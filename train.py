import argparse
import math
import os
import random

import numpy as np
import torch

from data import MovieLensDataset
from model import SASRec

from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# def sample_negative_items(pos_items: torch.Tensor, item_num: int, device: torch.device) -> torch.Tensor:
#     neg_items = torch.randint(1, item_num + 1, size=pos_items.shape, device=device)
#     # Keep padding aligned: if target position is padding (0), set negative to 0 too.
#     neg_items[pos_items == 0] = 0
#     return neg_items


# def sample_negative_items(pos_items,seq_items,item_num,device):
#     """
#     User-aware negative sampling (prefix-aware):
#     - For each user row, collect movies already seen in the input sequence (non-zero IDs).
#     - For each valid target position (pos_items != 0), sample a random movie ID
#       that is NOT in seen set and NOT equal to the positive target.
#     - Keep padding positions as 0.
#     """
#     # Start with all zeros; we will fill valid positions.
#     neg_items = torch.zeros_like(pos_items, device=device)
#     batch_size, seq_len = pos_items.shape
#     for i in range(batch_size):
#         # Movies already present in this user's input sequence prefix.
#         # We ignore 0 because 0 is padding, not a real movie.
#         seen = set(seq_items[i][seq_items[i] != 0].tolist())
#         for j in range(seq_len):
#             # If this target position is padding, keep negative as 0.
#             if pos_items[i, j].item() == 0:
#                 neg_items[i, j] = 0
#                 continue
#             pos_id = pos_items[i, j].item()
#             # Re-sample until we get an item that is not already seen
#             # and not equal to the positive target.
#             while True:
#                 neg_id = random.randint(1, item_num)
#                 if (neg_id not in seen) and (neg_id != pos_id):
#                     neg_items[i, j] = neg_id
#                     break
#     return neg_items


def sample_negative_items(pos_items, user_histories_batch, item_num, device):
    """
    SASRec-style negative sampling:
    - For each user row and each valid target position:
      sample random item from [1, item_num]
      reject it if:
        (a) item is in that user's full interaction history
        (b) item equals current positive target
    - Keep padding positions (pos == 0) as 0.
    """
    neg_items = torch.zeros_like(pos_items, device=device)
    batch_size, seq_len = pos_items.shape
    for i in range(batch_size):
        seen_items = user_histories_batch[i]  # set of all items this user has interacted with
        for j in range(seq_len):
            pos_id = int(pos_items[i, j].item())
            # Padding position -> keep 0
            if pos_id == 0:
                neg_items[i, j] = 0
                continue
            # Rejection sampling until valid negative found
            while True:
                neg_id = random.randint(1, item_num)
                if (neg_id not in seen_items) and (neg_id != pos_id):
                    neg_items[i, j] = neg_id
                    break
    return neg_items





def parse_args():
    parser = argparse.ArgumentParser("Train SASRec on MovieLens 1M dataset")

    parser.add_argument("--movies_file", type=str, default="movies.dat")
    parser.add_argument("--ratings_file", type=str, default="ratings.dat")

    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="sasrec_best.pt")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)

    return parser.parse_args()


@torch.no_grad()
def validate(model, val_loader, dataset, topk=(10, 20)):
    model.eval()
    device = model.device

    recall_sum = {}
    ndcg_sum = {}
    for k in topk:
        recall_sum[k] = 0.0
        ndcg_sum[k] = 0.0
    num_users = 0

    for seq, target, row_idx in val_loader:
        seq = seq.to(device)
        target = target.to(device)

        # Get scores for all items.
        hidden = model(seq)
        logits = model.predict_all_items(hidden)  # [batch_size, item_num + 1]
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        # Do not rank padding id=0.
        logits[:, 0] = -1e9

        # Full-ranking protocol: mask items already in user's training history
        # so ranking happens among items the user hasn't seen in training.
        for i, r in enumerate(row_idx.tolist()):
            seen = dataset.train_histories[r]
            if seen:
                logits[i, list(seen)] = -1e9

        # Score of true target item for each user.
        target_scores = logits.gather(1, target.unsqueeze(1))  # [batch_size, 1]

        # Rank = how many items have score >= true item score (1 is best).
        ranks = (logits >= target_scores).sum(dim=1)
        ranks = torch.clamp(ranks, min=1)

        for k in topk:
            hits = (ranks <= k).float()

            # Recall@K
            recall_sum[k] += hits.sum().item()

            # NDCG@K
            dcg = hits / torch.log2(ranks.float() + 1.0)
            ndcg_sum[k] += dcg.sum().item()

        num_users += seq.size(0)

    metrics = {}
    for k in topk:
        metrics[f"recall@{k}"] = recall_sum[k] / num_users
        metrics[f"ndcg@{k}"] = ndcg_sum[k] / num_users
    return metrics


def train(args):
    set_seed(args.seed)

    # Load data.
    dataset = MovieLensDataset(
        movies_file=args.movies_file,
        ratings_file=args.ratings_file,
        maxlen=args.maxlen,
    )

    # Build train loader with row indices so each batch row can map to user history.
    # This lets us use the correct full user history set for negative sampling.
    train_X = torch.tensor(dataset.splits["train"][0], dtype=torch.int64)
    train_y = torch.tensor(dataset.splits["train"][1], dtype=torch.int64)
    train_idx = torch.arange(train_X.size(0), dtype=torch.int64)  # row id of each training sample
    train_loader = DataLoader(
        TensorDataset(train_X, train_y, train_idx),  # returns (seq, pos, row_idx)
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Build val loader with row indices so each batch row can map to its user's training history.
    # Needed for seen-item masking during full-ranking evaluation.
    val_X = torch.tensor(dataset.splits["val"][0], dtype=torch.int64)
    val_y = torch.tensor(dataset.splits["val"][1], dtype=torch.int64)
    val_idx = torch.arange(val_X.size(0), dtype=torch.int64)
    val_loader = DataLoader(
        TensorDataset(val_X, val_y, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Number of items (excluding padding id 0).
    item_num = int(dataset.movies["MovieID"].max())

    # Build model.
    model = SASRec(
        item_num=item_num,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        lr=args.lr,
    )

    device = model.device

    # Linear warmup for first 5% of steps, then cosine decay to 0.
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda)

    # Prepare checkpoint folder/file.
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)

    # Track best validation NDCG@10.
    best_val_ndcg10 = -1.0
    wait = 0

    # Train loop.
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for seq, pos, row_idx in train_loader:
            # Move batch to same device as model.
            seq = seq.to(device)
            pos = pos.to(device)

            # Sample negatives with same shape as pos.
            # Get full interacted-item set for each row/user in this batch.
            batch_histories = [dataset.user_histories[i] for i in row_idx.tolist()]
            neg = sample_negative_items(
                pos_items=pos,
                user_histories_batch=batch_histories,  # full user history exclusion
                item_num=item_num,
                device=device,
            )

            # Forward pass.
            hidden = model(seq)

            # Score positive and negative items.
            pos_scores = model.predict_next(hidden, pos)
            neg_scores = model.predict_next(hidden, neg)

            # Ignore padding in loss.
            mask = (pos != 0)

            # Compute loss.
            loss = model.compute_loss(pos_scores, neg_scores, mask)

            # Backprop + update.
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            model.optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)

        val_metrics = validate(model, val_loader, dataset, topk=(10, 20))
        val_ndcg10 = val_metrics["ndcg@10"]

        print("Epoch", epoch)
        print("  train_loss   =", round(avg_loss, 4))
        print("  val_recall@10=", round(val_metrics["recall@10"], 4))
        print("  val_recall@20=", round(val_metrics["recall@20"], 4))
        print("  val_ndcg@10  =", round(val_metrics["ndcg@10"], 4))
        print("  val_ndcg@20  =", round(val_metrics["ndcg@20"], 4))

        # Save best model based on val NDCG@10.
        if val_ndcg10 > best_val_ndcg10:
            best_val_ndcg10 = val_ndcg10
            wait = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_ndcg10": best_val_ndcg10,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print("  saved best checkpoint:", ckpt_path)
        else:
            wait += 1
            print("  no improvement, wait =", wait)
            if wait >= args.patience:
                print("Early stopping triggered.")
                break

    print("Training done.")
    print("Best val NDCG@10 =", round(best_val_ndcg10, 4))


if __name__ == "__main__":
    args = parse_args()
    train(args)
