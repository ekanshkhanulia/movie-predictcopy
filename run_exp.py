import argparse
import csv
import itertools # for iterating over combinations of hyperparameters
import os
import re
import subprocess # Runs train.py and evaluate.py as child processes.
import sys # Gives Python executable path (sys.executable).

def parse_args():
    parser = argparse.ArgumentParser("Run SASRec ablation experiments")  # Creates parser for experiment runner options.
    
    parser.add_argument("--movies_file", type=str, default="movies.dat")  # Path to MovieLens movies file.
    parser.add_argument("--ratings_file", type=str, default="ratings.dat")  # Path to MovieLens ratings file.
    
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")  # Folder to store checkpoints from each run.
    parser.add_argument("--result_dir", type=str, default="results")  # Folder to store experiment CSV/results.
    parser.add_argument("--result_csv", type=str, default="focused_grid_results.csv")  # Output CSV filename. (focused grid: 3 archs x 2 dropouts x 2 LRs)
    
    parser.add_argument("--epochs", type=int, default=50)  # Epoch count passed to train.py.
    parser.add_argument("--batch_size", type=int, default=128)  # Batch size passed to train.py/evaluate.py.
    parser.add_argument("--patience", type=int, default=5)  # Early stopping patience passed to train.py.
    parser.add_argument("--lr", type=float, default=1e-3)  # Learning rate passed to train.py.
    parser.add_argument("--dropout_rate", type=float, default=0.2)  # Dropout passed to train.py.

    # change after grid search - K negatives per position. Default 1 preserves
    # the previous (single-negative) behavior. See CHANGE_LOG Chapter 12.
    parser.add_argument("--num_negatives", type=int, default=1)

    return parser.parse_args()  # Parses args and returns args object.


def run_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)  # Executes command and captures stdout/stderr as text.
    return result.returncode  # Returns process exit code (0 means success, non-zero means failure).


def parse_eval_metrics(output_text):
    metrics = {
        "val_recall@10": None,
        "val_recall@20": None,
        "val_ndcg@10": None,
        "val_ndcg@20": None,
        "test_recall@10": None,
        "test_recall@20": None,
        "test_ndcg@10": None,
        "test_ndcg@20": None,
    }

    mode = None
    for raw_line in output_text.splitlines():
        line = raw_line.strip()

        if line.startswith("Validation Metrics"):
            mode = "val"
            continue
        if line.startswith("Test Metrics"):
            mode = "test"
            continue

        if mode is None:
            continue

        m = re.search(r"(Recall@10|Recall@20|NDCG@10|NDCG@20)\s*=\s*([0-9]*\.?[0-9]+)", line)
        if not m:
            continue

        metric_name = m.group(1).lower()  # recall@10, ndcg@20, etc.
        metric_value = float(m.group(2))
        key = f"{mode}_{metric_name}"
        metrics[key] = metric_value

    return metrics



def main(args):
    # FOCUSED GRID — three architectures (full-grid top-3) crossed with
    # dropout x lr sweep. Total = 3 x 2 x 2 = 12 runs. Replaces the original
    # 81-cell architectural grid; results land in focused_grid_results.csv.
    architectures = [
        {"num_blocks": 2, "hidden_units": 128, "num_heads": 4, "maxlen": 50},  # rank 1 (val NDCG@10 = 0.026538)
        {"num_blocks": 2, "hidden_units": 128, "num_heads": 2, "maxlen": 50},  # rank 2 (val NDCG@10 = 0.026366)
        {"num_blocks": 3, "hidden_units": 128, "num_heads": 4, "maxlen": 50},  # rank 3 (val NDCG@10 = 0.026191)
    ]
    dropouts = [0.1, 0.2]
    lrs = [0.001, 0.005]

    os.makedirs(args.ckpt_dir, exist_ok=True)  # Ensures checkpoint folder exists.
    os.makedirs(args.result_dir, exist_ok=True)  # Ensures result folder exists.
    result_csv_path = os.path.join(args.result_dir, args.result_csv)  # Builds full path for CSV output file.

    # Build all (arch, dropout, lr) combinations explicitly so each row has
    # all three swept values recorded.
    all_combinations = []
    for arch in architectures:
        for dropout in dropouts:
            for lr in lrs:
                cfg = dict(arch)
                cfg["dropout_rate"] = dropout
                cfg["lr"] = lr
                all_combinations.append(cfg)
    total_runs = len(all_combinations)

    # Check which run_ids are already completed in the CSV (resume support).
    completed_ids = set()
    if os.path.exists(result_csv_path):
        with open(result_csv_path, "r", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            for row in reader:
                try:
                    completed_ids.add(int(row["run_id"]))
                except (KeyError, ValueError):
                    pass
        print(f"Resuming: {len(completed_ids)} runs already done, skipping them.")
        file_mode = "a"  # Append to existing CSV.
    else:
        file_mode = "w"  # Fresh start, write new CSV.

    with open(result_csv_path, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if file_mode == "w":  # Only write header when starting fresh.
            writer.writerow([
                "run_id",
                "num_blocks",
                "hidden_units",
                "num_heads",
                "maxlen",
                "dropout_rate",
                "lr",
                "train_exit_code",
                "eval_exit_code",
                "val_recall@10",
                "val_recall@20",
                "val_ndcg@10",
                "val_ndcg@20",
                "test_recall@10",
                "test_recall@20",
                "test_ndcg@10",
                "test_ndcg@20",
            ])

        run_id = 0  # Initializes run counter.

        for cfg in all_combinations:  # Iterates through each (arch, dropout, lr) combination.
            run_id += 1  # Increments run number.

            if run_id in completed_ids:  # Skip already completed runs.
                print(f"Run {run_id}/{total_runs} already done, skipping.")
                continue

            print(
                f"Run {run_id}/{total_runs} | "
                f"num_blocks={cfg['num_blocks']}, "
                f"hidden_units={cfg['hidden_units']}, "
                f"num_heads={cfg['num_heads']}, "
                f"maxlen={cfg['maxlen']}, "
                f"dropout={cfg['dropout_rate']}, "
                f"lr={cfg['lr']}"
            )

            ckpt_name = f"sasrec_run_{run_id}.pt"  # Creates unique checkpoint filename per run.

            train_cmd = [
                    sys.executable,  # Uses current Python interpreter path (safe across environments).
                    "train.py",  # Calls your training script.
                    "--movies_file", args.movies_file,  # Passes movies file path to training.
                    "--ratings_file", args.ratings_file,  # Passes ratings file path to training.
                    "--ckpt_dir", args.ckpt_dir,  # Folder where checkpoint should be saved.
                    "--ckpt_name", ckpt_name,  # Unique checkpoint name for this run.
                    "--epochs", str(args.epochs),  # Number of epochs for this run.
                    "--batch_size", str(args.batch_size),  # Batch size for this run.
                    "--patience", str(args.patience),  # Early stopping patience for this run.
                    "--lr", str(cfg["lr"]),  # Per-run learning rate (focused grid axis).
                    "--dropout_rate", str(cfg["dropout_rate"]),  # Per-run dropout (focused grid axis).
                    "--num_blocks", str(cfg["num_blocks"]),  # Current combo: transformer blocks.
                    "--hidden_units", str(cfg["hidden_units"]),  # Current combo: hidden size.
                    "--num_heads", str(cfg["num_heads"]),  # Current combo: attention heads.
                    "--maxlen", str(cfg["maxlen"]),  # Current combo: max sequence length.
                    "--num_negatives", str(args.num_negatives),  # change after grid search - K negatives passthrough
                ]

            train_exit_code = run_command(train_cmd)

            eval_cmd = [
                    sys.executable,  # Current Python interpreter.
                    "evaluate.py",  # Evaluation script.
                    "--movies_file", args.movies_file,  # Same movies file.
                    "--ratings_file", args.ratings_file,  # Same ratings file.
                    "--ckpt_path", os.path.join(args.ckpt_dir, ckpt_name),  # Checkpoint produced by current run.
                    "--batch_size", str(args.batch_size),  # Eval batch size.
                    "--maxlen", str(cfg["maxlen"]),  # Same maxlen as training config.
                ]
            parsed_metrics = {
                "val_recall@10": None,
                "val_recall@20": None,
                "val_ndcg@10": None,
                "val_ndcg@20": None,
                "test_recall@10": None,
                "test_recall@20": None,
                "test_ndcg@10": None,
                "test_ndcg@20": None,
            }
            if train_exit_code == 0:  # Only evaluate if training succeeded.
                eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
                eval_exit_code = eval_result.returncode
                if eval_exit_code == 0:
                    parsed_metrics = parse_eval_metrics(eval_result.stdout)
            else:
                eval_exit_code = -1  # Mark eval as skipped/failed if training failed.

            writer.writerow([  # Writes one experiment result row.
                run_id,  # Run index number.
                cfg["num_blocks"],  # Blocks used in this run.
                cfg["hidden_units"],  # Hidden size used in this run.
                cfg["num_heads"],  # Attention heads used in this run.
                cfg["maxlen"],  # Max sequence length used in this run.
                cfg["dropout_rate"],  # Dropout used in this run (focused grid axis).
                cfg["lr"],  # Learning rate used in this run (focused grid axis).
                train_exit_code,  # 0 means train success, non-zero means failure.
                eval_exit_code,  # 0 means eval success, -1 means skipped/failed.
                parsed_metrics["val_recall@10"],
                parsed_metrics["val_recall@20"],
                parsed_metrics["val_ndcg@10"],
                parsed_metrics["val_ndcg@20"],
                parsed_metrics["test_recall@10"],
                parsed_metrics["test_recall@20"],
                parsed_metrics["test_ndcg@10"],
                parsed_metrics["test_ndcg@20"],
            ])
            f.flush()
            if eval_exit_code == 0:
                print(
                    "  Val  | "
                    f"Recall@10={parsed_metrics['val_recall@10']}, "
                    f"Recall@20={parsed_metrics['val_recall@20']}, "
                    f"NDCG@10={parsed_metrics['val_ndcg@10']}, "
                    f"NDCG@20={parsed_metrics['val_ndcg@20']}"
                )
                print(
                    "  Test | "
                    f"Recall@10={parsed_metrics['test_recall@10']}, "
                    f"Recall@20={parsed_metrics['test_recall@20']}, "
                    f"NDCG@10={parsed_metrics['test_ndcg@10']}, "
                    f"NDCG@20={parsed_metrics['test_ndcg@20']}"
                )
            else:
                print("  Metrics unavailable (train/eval failed for this run).")
            print(f"Run {run_id}/{total_runs} done")
    print("\nAll experiment runs finished.")  # Prints completion message after loop ends.
    print("Results CSV saved at:", result_csv_path)  # Prints where summary CSV is stored.
if __name__ == "__main__":  # Runs only when this file is executed directly.
    args = parse_args()  # Reads command-line args (or defaults).
    main(args)  # Starts full experiment pipeline.

    