import torch
from torch import nn
import torch.nn.functional as F

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.GELU(), # Smoother than ReLU
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(self, X):
        return self.model(X)

class SASRec(nn.Module):
    def __init__(self, item_num, maxlen, hidden_units, num_blocks, num_heads, dropout_rate, lr):
        super().__init__()

        # Item embeddings
        # item_num + 1 because index 0 is reserved for padding
        # padding_idx=0 ensures pad tokens always produce a zero vector
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0) 

        # Positional embeddings
        self.pos_emb = nn.Embedding(maxlen, hidden_units)

        # Dropout on item + position embeddings for regularization
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Transformer blocks, Pre-LN style
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_units,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True,
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        #self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)  change 1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.0)

        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, X):
        """
        Passes the input sequence through the embedding layer and Transformer blocks, returning a hidden state at each position.
        """
        seq_len = X.shape[1]

        # # Create position indices [0, 1, ..., seq_len-1] for each sequence in batch
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand_as(X)

        # Combine item and positional embeddings, and add dropout
        # change after grid search - DOESN'T WORK
        # Tried scaling item embeddings by sqrt(d) to match official SASRec
        # (kang205/SASRec modules.py, scale=True) and Vaswani et al. 2017.
        # Result: val NDCG@10 0.026540 -> 0.026300 (-0.9%); 7/8 metrics down.
        # Likely cause: official uses TF init (small std ~0.01), PyTorch
        # nn.Embedding defaults to N(0,1) - scaling by sqrt(128) ~= 11.31 on
        # top of that overwhelms the positional signal. Reverted.
        # See CHANGE_LOG Chapter 11.
        # seqs = self.item_emb(X) * (self.item_emb.embedding_dim ** 0.5) + self.pos_emb(positions)
        seqs = self.item_emb(X) + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Zero out padding positions so they don't influence attention 
        seqs *= (X != 0).unsqueeze(-1)
        # ignore padding positions
        key_padding_mask = (X == 0)

        # Causal attention mask: enforces each position to only see past items
        # the upper triangle is -inf so pos t cannot see t+1, t+2...
        attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=self.device),diagonal=1)

        for i in range(len(self.attention_layers)):
            # Pre-LN: normalize before attention, then add residual
            seqs_norm = self.attention_layernorms[i](seqs)

            attn_output, _ = self.attention_layers[i](
                seqs_norm, seqs_norm, seqs_norm,
                attn_mask=attn_mask,
                #key_padding_mask=key_padding_mask  #chnage2
            )

            # Residual connection
            seqs = seqs + attn_output

            # Nnormalize before FFN, then add residual
            seqs_norm = self.forward_layernorms[i](seqs)
            ffn_output = self.forward_layers[i](seqs_norm)

            seqs = seqs + ffn_output

        return self.last_layernorm(seqs)
    
    def predict_next(self, seq_hidden, item_indices):
        """
        Prediction layer used during training: scores one item per position via dot product 
        between the hidden state and the item's embedding.
        Used to score both positive (true next) and negative (sampled) items.
        """
        item_embs = self.item_emb(item_indices)
        return (seq_hidden * item_embs).sum(dim=-1)

    def predict_all_items(self, seq_hidden):
        """
        Prediction layer used at inference: scores all items using the last position's hidden state.
        Used to rank all items.
        """
        h = seq_hidden[:, -1, :]
        all_embs = self.item_emb.weight
        return torch.matmul(h, all_embs.T)
    
    def compute_loss(self, pos_scores, neg_scores, mask):
        """
        Binary cross-entropy loss with negative sampling.

        Supports both single-negative and multi-negative cases:
          - single:  pos_scores (B, T), neg_scores (B, T)
          - multi :  pos_scores (B, T), neg_scores (B, T, K)

        For the multi-negative case BCE is averaged over the trailing K
        dimension so the per-position scale of the negative term matches the
        positive term and the overall loss magnitude stays comparable to the
        K=1 case.

        Padding positions are excluded from the loss using the mask (which is
        True at non-padding positions).

        change after grid search - K negatives support. When K=1 this reduces
        to the original single-negative computation. See CHANGE_LOG Chapter 12.
        """
        # Positive items are labeled 1, negatives are labeled 0
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores), reduction='none'
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores), reduction='none'
        )

        # change after grid search - if neg_scores has an extra K dim, average
        # over it so the per-position negative loss has the same shape as
        # pos_loss. For K=1 this is mean over a singleton dim (no-op).
        if neg_loss.dim() == pos_loss.dim() + 1:
            neg_loss = neg_loss.mean(dim=-1)

        loss = (pos_loss + neg_loss)[mask]
        return loss.mean()

