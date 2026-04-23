import argparse
import math #for ndcg
import torch

from data import MovieLensDataset
from model import SASRec


#step
def parse_args():
    parser = argparse.ArgumentParser("Evaluate trained SASRec model")  # Creates CLI parser for evaluation script.
    parser.add_argument("--movies_file", type=str, default="movies.dat")  # Path to Moviw file.
    parser.add_argument("--ratings_file", type=str, default="ratings.dat")  # Path to ratings interactions file.
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/sasrec_best.pt")  # Path to saved checkpoint from training.
    
    parser.add_argument("--batch_size", type=int, default=256)  # Evaluation batch size (usually can be larger than training).
    parser.add_argument("--maxlen", type=int, default=50)  # Sequence length used when rebuilding dataset for eval.
    
    return parser.parse_args()  # Parses CLI args and returns them as args object.

#step
def recall_at_k(rank, k):
    if rank <= k:  # If true item is inside top-k predicted items, it's a hit.
        return 1.0  # Recall contribution for this user is 1.
    return 0.0  # Otherwise true item is outside top-k, contribution is 0.
def ndcg_at_k(rank, k):
    if rank > k:  # If true item is not in top-k, NDCG contribution is 0.
        return 0.0
    return 1.0 / math.log2(rank + 1)  # If in top-k, reward higher ranks more (rank 1 gets highest score).


def format_trunc_6(x):
    factor = 10 ** 6
    y = math.trunc(float(x) * factor) / factor
    return f"{y:.6f}"


#step 
def build_model_from_checkpoint(ckpt_path,dataset):
    ckpt=torch.load(ckpt_path,map_location="cpu") # Loads saved checkpoint dictionary from disk to CPU memory.
    saved_args=ckpt["args"] # Reads training-time hyperparameters stored during checkpoint save.

    item_num=int(dataset.movies["MovieID"].max()) # Recomputes total number of items from current dataset.


    model=SASRec(
        # Rebuilds SASRec architecture using same hyperparameters used during training.
        item_num=item_num,
        maxlen=saved_args["maxlen"],
        hidden_units=saved_args["hidden_units"],
        num_blocks=saved_args["num_blocks"],
        num_heads=saved_args["num_heads"],
        dropout_rate=saved_args["dropout_rate"],
        lr=saved_args["lr"],
        )
    
    
    model.load_state_dict(ckpt["model_state_dict"])  # Loads learned weights into the rebuilt model.
    model.eval()  # Puts model into evaluation mode (disables dropout behavior).
    return model, item_num  # Returns ready model and item count for evaluation use.


#step
@torch.no_grad()
def evaluate_split(model, loader, item_num, topk=(10, 20)):
    model.eval()
    device = model.device

    recall_total = {}
    ndcg_total = {}
    for k in topk:
        recall_total[k] = 0.0
        ndcg_total[k] = 0.0

    user_count = 0
    _ = item_num  # keeps signature explicit for compatibility/readability

    for seq, target in loader:
        seq = seq.to(device)
        target = target.to(device)

        hidden = model(seq)
        logits = model.predict_all_items(hidden)  # [batch_size, item_num + 1]
        logits[:, 0] = -1e9  # never rank padding id

        sorted_items = torch.argsort(logits, dim=1, descending=True)

        for i in range(seq.size(0)):
            true_item = target[i].item()
            rank_index = (sorted_items[i] == true_item).nonzero(as_tuple=False).item()
            rank = rank_index + 1

            for k in topk:
                recall_total[k] += recall_at_k(rank, k)
                ndcg_total[k] += ndcg_at_k(rank, k)

            user_count += 1

    metrics = {}
    for k in topk:
        metrics[f"recall@{k}"] = recall_total[k] / user_count
        metrics[f"ndcg@{k}"] = ndcg_total[k] / user_count
    return metrics


#step
def main(args):
    dataset= MovieLensDataset(
        args.movies_file,
        args.ratings_file, 
        args.maxlen)


    val_loader=dataset.get_loader("val",args.batch_size) #validation Data Loader
    test_loader=dataset.get_loader("test",args.batch_size) #test Data Loader

    model,item_num=build_model_from_checkpoint(args.ckpt_path,dataset) #rebuilds model 

    val_metrics=evaluate_split(model,val_loader,item_num=item_num,topk=(10,20))
    test_metrics=evaluate_split(model,test_loader,item_num=item_num,topk=(10,20))


    print("Validation Metrics:")
    print("Recall@10=", format_trunc_6(val_metrics["recall@10"]))
    print("  Recall@20 =", format_trunc_6(val_metrics["recall@20"]))  # Prints validation Recall@20.
    print("  NDCG@10   =", format_trunc_6(val_metrics["ndcg@10"]))  # Prints validation NDCG@10.
    print("  NDCG@20   =", format_trunc_6(val_metrics["ndcg@20"]))  # Prints validation NDCG@20.
    print("Test Metrics")  # Header for test output.
    print("  Recall@10 =", format_trunc_6(test_metrics["recall@10"]))  # Prints test Recall@10.
    print("  Recall@20 =", format_trunc_6(test_metrics["recall@20"]))  # Prints test Recall@20.
    print("  NDCG@10   =", format_trunc_6(test_metrics["ndcg@10"]))  # Prints test NDCG@10.
    print("  NDCG@20   =", format_trunc_6(test_metrics["ndcg@20"]))  # Prints test NDCG@20.

if __name__ == "__main__":  # Runs this block only when evaluate.py is executed directly.
    args = parse_args()  # Reads command-line arguments (or defaults) into args object.
    main(args)  # Starts full evaluation pipeline using parsed arguments.