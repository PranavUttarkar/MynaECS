import os
import pickle
import torch
from argparse import Namespace

# Import functions from your existing modules.
from utils import get_n_frames  # used in both pipelines
from train import setup_for_training, parse_args as parse_train_args
from evaluate import (
    compute_embeddings,
    load_embeds,
    save_embeds,
    get_hyperparameter_grid,
    evaluate,
    parse_args as parse_eval_args,
    get_dataset,
)
import warnings
warnings.filterwarnings('ignore')


def extract_dataset_embeddings(model, args):
    """
    Compute embeddings for each dataset split and return a dictionary with
    keys 'train', 'valid', 'test'.
    """
    embeddings_dict = {}
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(args.dataroot, split)
        print(f"[+] Loading {split} dataset from: {split_path}")
        dataset, _ = get_dataset(dataroot=split_path, args=args)
        # Remove frame-size constraint if present
        dataset.frame_size = None
        testmode = (split != "train")
        print(f"[+] Computing embeddings for {split} set...")
        embeds_dataset = compute_embeddings(model, dataset, args, testmode=testmode)
        embeddings_dict[split] = {
            "indices": embeds_dataset.indices.cpu(),
            "embeddings": embeds_dataset.embeddings.cpu(),
            "labels": embeds_dataset.labels,
        }
    return embeddings_dict


def save_all_embeddings(embeddings_dict, filepath="embeddings.pkl"):
    """Save the computed embeddings to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(embeddings_dict, f)
    print(f"[+] Saved embeddings to {filepath}")
    return filepath


def run_evaluation(args, embed_filepath):
    """
    Runs the grid search evaluation by loading the precomputed embeddings.
    """
    # Load embeddings from file and override datasets in evaluation routine.
    print("[+] Loading embeddings for evaluation...")
    train_dataset, valid_dataset, test_dataset = load_embeds(embed_filepath)

    hyperparam_grid = get_hyperparameter_grid()
    print(f"==> Starting grid search over {len(hyperparam_grid)} hyperparameter combinations.")

    best_overall_metric = -float("inf")
    best_overall_metrics = {}
    best_hyperparams = None

    # Loop over hyperparameter combinations (grid search)
    for idx, hyperparams in enumerate(hyperparam_grid):
        # (Optional) Skip some indices for fast evaluation, if desired.
        if args.fast_eval and (idx + 1) not in [2, 3, 4, 5, 6, 9, 10, 11, 12, 
                                                   13, 14, 16, 19, 37, 43, 52, 55, 
                                                   56, 58, 59, 70, 94, 106, 213]:
            continue
        print(f"==> Starting run {idx + 1}/{len(hyperparam_grid)}")
        best_primary_metric, best_metrics = evaluate(train_dataset, valid_dataset, test_dataset, hyperparams, args)
        if best_primary_metric > best_overall_metric:
            best_overall_metric = best_primary_metric
            best_overall_metrics = best_metrics.copy()
            best_hyperparams = hyperparams

    print("\n==> Grid Search Complete.")
    print("Best Overall Results:")
    for metric_name, value in best_overall_metrics.items():
        if metric_name != 'text':
            print(f"  {metric_name}: {value}")
    print("\nBest Hyperparameters:")
    print(best_hyperparams)
    if args.eval_save_results:
        with open(args.eval_save_results, 'w') as f:
            import json
            json.dump(best_overall_metrics, f, indent=4)
        print(f"[+] Saved evaluation results to {args.eval_save_results}")


def main():
    """
    Main pipeline:
    1. Load the model and datasets.
    2. Compute embeddings.
    3. Save embeddings.
    4. Run evaluation grid search.
    """
    # Parse arguments from evaluation pipeline.
    # (Assumes your evaluate.py and train.py share similar CLI arguments)
    args = parse_eval_args()
    # Ensure device is not CPU (if required)
    assert args.device.lower() != 'cpu', "CUDA must be available for this pipeline."

    # Setup model and training components (loads model weights, etc.)
    print("[1/3] Setting up model and training environment...")
    model, _, _, train_loader, test_dataset, use_wandb, wandb = setup_for_training(rank=0, world_size=1, args=args)
    model.linear_head = torch.nn.Identity()  # Remove classification head

    # Extract embeddings for each dataset split
    print("[2/3] Extracting embeddings from dataset splits...")
    embeddings_dict = extract_dataset_embeddings(model, args)
    embed_filepath = save_all_embeddings(embeddings_dict, filepath=args.eval_load_embeds or "embeddings.pkl")

    # Run evaluation (grid search)
    print("[3/3] Running evaluation grid search on embeddings...")
    run_evaluation(args, embed_filepath)


if __name__ == '__main__':
    main()
