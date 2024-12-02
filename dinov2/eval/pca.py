import argparse
import torch
import numpy as np
from utils import PCA, IncrementalPCAWrapper
from dinov2.data import make_data_loader, make_dataset
from dinov2.eval.metrics import build_topk_accuracy_metric, AccuracyAveraging
from dinov2.eval.utils import extract_features
import json
from torch.nn.functional import softmax
from dinov2.data.transforms import (
    make_classification_eval_transform,
)
import logging

logger = logging.getLogger("dinov2")

def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:  # Convert single-value tensor to a Python scalar
            return obj.item()
        else:  # Convert multi-value tensor to a Python list
            return obj.tolist()
    elif isinstance(obj, dict):  # Recursively process dictionaries
        return {key: tensor_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):  # Recursively process lists
        return [tensor_to_python(item) for item in obj]
    return obj  # Return the object as is if not a tensor

def get_args_parser():
    parser = argparse.ArgumentParser(description="PCA-based evaluation")
    parser.add_argument("--train-dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("--val-dataset", type=str, required=True, help="Validation dataset")
    parser.add_argument("--num-components", type=int, required=True, help="Number of PCA components")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for feature extraction")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--nb-knn", nargs="+", type=int, default=[10, 20, 100, 200], help="Number of nearest neighbors")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for k-NN voting")
    parser.add_argument("--n-per-class-list", nargs="+", type=int, default=[-1], help="Samples per class")
    parser.add_argument("--n-tries", type=int, default=1, help="Number of trials")
    return parser


def extract_pca_features(dataset, batch_size, num_workers, num_components, device="cuda"):
    """
    Extract features from a dataset and apply Incremental PCA to reduce dimensionality.
    :param dataset: Dataset object with images and labels.
    :param batch_size: Batch size for processing.
    :param num_workers: Number of workers for the DataLoader.
    :param num_components: Number of PCA components to reduce to.
    :param device: Device to use ('cuda' or 'cpu').
    :return: PCA-reduced features and labels.
    """
    # Prepare DataLoader
    dataloader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=None,
        drop_last=False,
        shuffle=False,
    )

    all_features, all_labels = [], []

    # Initialize IncrementalPCA
    ipca = IncrementalPCAWrapper(num_components=num_components, batch_size=batch_size)

    for samples, labels in dataloader:
        samples = samples.to(device)
        features = samples.view(samples.size(0), -1)  # Flatten images if needed
        
        # Apply Incremental PCA in batches
        if len(all_features) == 0:
            # First batch, just fit and transform
            ipca.fit(features)
        reduced_features = ipca.transform(features)
        
        # Store features and labels
        all_features.append(reduced_features)
        all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"PCA-reduced features shape: {all_features.shape}, labels shape: {all_labels.shape}.")
    return all_features, all_labels

def eval_knn_pca_chunk(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    nb_knn: list,
    temperature: float,
    n_per_class_list: list = [-1],
    n_tries: int = 1,
    device: str = "cuda",
    output_dir: str = "./results",  # Directory to store logs
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting k-NN evaluation with PCA features on device {device}.")
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    num_classes = int(train_labels.max() + 1)
    metric_collection = build_topk_accuracy_metric(
        average_type=AccuracyAveraging.MEAN_ACCURACY,
        num_classes=num_classes,
    )
    results_dict = {}
    all_predictions = []  # To store predictions and actuals

    for npc in n_per_class_list:
        for t in range(n_tries):
            if npc >= 0:
                torch.manual_seed(t)
                indices = []
                for cls in range(num_classes):
                    cls_indices = (train_labels == cls).nonzero(as_tuple=True)[0]
                    sampled_indices = cls_indices[torch.randperm(len(cls_indices))[:npc]]
                    indices.append(sampled_indices)
                indices = torch.cat(indices)
                sampled_train_features = train_features[indices]
                sampled_train_labels = train_labels[indices]
            else:
                sampled_train_features = train_features
                sampled_train_labels = train_labels

            for k in nb_knn:
                print(f"Evaluating k={k} neighbors for {npc} samples per class, try={t+1}...")

                similarities = torch.mm(val_features, sampled_train_features.T)
                topk_sims, topk_indices = similarities.topk(k, dim=1)
                topk_labels = sampled_train_labels[topk_indices]
                topk_sims = softmax(topk_sims / temperature, dim=1)
                class_votes = torch.zeros(val_features.size(0), num_classes, device=device)
                for i in range(k):
                    class_votes.scatter_add_(1, topk_labels[:, i].unsqueeze(1), topk_sims[:, i].unsqueeze(1))

                preds = class_votes.argmax(dim=1)
                preds = preds.to(device)
                val_labels = val_labels.to(device)

                # Log predictions and actual labels
                pred_actual_pairs = [{"prediction": p.item(), "actual": a.item()} for p, a in zip(preds, val_labels)]
                all_predictions.extend(pred_actual_pairs)

                # Metrics
                accuracy = (preds == val_labels).float().mean().item()
                print(f"k={k}, accuracy={accuracy:.4f}")

                metric_key = (npc, t, k)
                metrics = metric_collection.clone()
                metrics.update(preds=class_votes, target=val_labels)
                results_dict[metric_key] = metrics.compute()

    # Save predictions and distribution logs
    with open(f"{output_dir}/predictions.json", "w") as f:
        json.dump(all_predictions, f, indent=4)

    # Calculate and save overall distribution
    pred_counts = {int(k): v for k, v in zip(*torch.unique(torch.tensor([x['prediction'] for x in all_predictions]), return_counts=True))}
    actual_counts = {int(k): v for k, v in zip(*torch.unique(torch.tensor([x['actual'] for x in all_predictions]), return_counts=True))}

    with open(f"{output_dir}/distribution.json", "w") as f:
        json.dump({"predictions": pred_counts, "actuals": actual_counts}, f, indent=4)

    print(f"Predictions and distribution saved in {output_dir}.")
    return results_dict



def eval_knn_pca(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    nb_knn: list,
    temperature: float,
    n_per_class_list: list = [-1],
    n_tries: int = 1,
    device: str = "cuda",
):
    """
    Perform k-NN evaluation on PCA-reduced features.
    :param train_features: Tensor of training features (num_train_samples, num_components)
    :param train_labels: Tensor of training labels (num_train_samples,)
    :param val_features: Tensor of validation features (num_val_samples, num_components)
    :param val_labels: Tensor of validation labels (num_val_samples,)
    :param nb_knn: List of k values to evaluate.
    :param temperature: Temperature parameter for softmax.
    :param n_per_class_list: Number of samples per class to use (-1 for all).
    :param n_tries: Number of sampling tries when n_per_class_list is specified.
    :param device: Device to perform computations on.
    :return: Dictionary with evaluation metrics for each k.
    """
    print(f"Starting k-NN evaluation with PCA features on device {device}.")
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    print(f"Training features: {train_features.shape}, validation features: {val_features.shape}")
    print("device", device, "train_features", train_features.device, "train_labels", train_labels.device, "val_features", val_features.device, "val_labels", val_labels.device)

    num_classes = int(train_labels.max() + 1)
    metric_collection = build_topk_accuracy_metric(
        average_type=AccuracyAveraging.MEAN_ACCURACY,
        num_classes=num_classes,
    )
    results_dict = {}

    for npc in n_per_class_list:
        for t in range(n_tries):
            # Filter training data if necessary
            if npc >= 0:
                torch.manual_seed(t)
                indices = []
                for cls in range(num_classes):
                    cls_indices = (train_labels == cls).nonzero(as_tuple=True)[0]
                    sampled_indices = cls_indices[torch.randperm(len(cls_indices))[:npc]]
                    indices.append(sampled_indices)
                indices = torch.cat(indices)
                sampled_train_features = train_features[indices]
                sampled_train_labels = train_labels[indices]
            else:
                sampled_train_features = train_features
                sampled_train_labels = train_labels

            for k in nb_knn:
                print(f"Evaluating k={k} neighbors for {npc} samples per class, try={t+1}...")

                # Compute similarities
                similarities = torch.mm(val_features, sampled_train_features.T)
                topk_sims, topk_indices = similarities.topk(k, dim=1)

                # Gather labels of nearest neighbors
                topk_labels = sampled_train_labels[topk_indices]

                # Compute probabilities
                topk_sims = softmax(topk_sims / temperature, dim=1)
                class_votes = torch.zeros(val_features.size(0), num_classes, device=device)
                for i in range(k):
                    class_votes.scatter_add_(1, topk_labels[:, i].unsqueeze(1), topk_sims[:, i].unsqueeze(1))

                # Predictions and metrics
                preds = class_votes.argmax(dim=1)

                # Ensure that predictions and targets are on the same device for metric computation
                preds = preds.to(device)  # Ensure preds are on the same device
                val_labels = val_labels.to(device)  # Ensure val_labels are on the same device

                accuracy = (preds == val_labels).float().mean().item()
                print(f"k={k}, accuracy={accuracy:.4f}")

                # Collect metrics
                metric_key = (npc, t, k)
                metrics = metric_collection.clone()
                metrics.update(preds=class_votes, target=val_labels)
                results_dict[metric_key] = metrics.compute()

    return results_dict




def main():
    print("Starting PCA..")
    args = get_args_parser().parse_args()

    # Prepare datasets
    train_dataset = make_dataset(dataset_str=args.train_dataset,transform=make_classification_eval_transform(resize_size=256, crop_size=224))
    val_dataset = make_dataset(dataset_str=args.val_dataset,transform=make_classification_eval_transform(resize_size=256, crop_size=224))

    # Extract training and validation features with PCA
    train_features, train_labels = extract_pca_features(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_components=args.num_components,
        device="cuda"
    )

    val_features, val_labels = extract_pca_features(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_components=args.num_components,
        device="cuda"
    )

    # Run k-NN evaluation
    results_dict = eval_knn_pca_chunk(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        nb_knn=args.nb_knn,
        temperature=args.temperature,
        n_per_class_list=args.n_per_class_list,
        n_tries=args.n_tries,
        device="cuda",
        chunk_size=1000  # Process in chunks to avoid OOM
    )


        # Convert results_dict to a JSON-serializable format
    results_dict_str = {str(key): tensor_to_python(value) for key, value in results_dict.items()}

    # Save as JSON
    with open(f"{args.output_dir}/pca_knn_results.json", "w") as f:
        json.dump(results_dict_str, f, indent=4)

if __name__ == "__main__":
    main()
