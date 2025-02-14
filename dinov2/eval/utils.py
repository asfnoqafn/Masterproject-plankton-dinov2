# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from tensorboard.plugins import projector
import google.protobuf.text_format as text_format
import logging
from typing import Dict, Optional , List
import os
import torch
import numpy as np
from torch import nn
from torchmetrics import MetricCollection
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb


import dinov2.distributed as distributed
from dinov2.data import (
    DatasetWithEnumeratedTargets,
    SamplerType,
    make_data_loader,
)
from dinov2.logging import MetricLogger

logger = logging.getLogger("dinov2")

def save_embeddings(features, labels, output_dir, step=0, max_samples=100):
    """
    Save embeddings to a file for later analysis, optionally limiting the number of samples.
    
    Args:
    features (torch.Tensor): Feature vectors to save
    labels (torch.Tensor): Corresponding labels for the features
    output_dir (str): Directory to save embeddings
    step (int, optional): Training step or epoch for filename
    max_samples (int, optional): Maximum number of samples to save
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # If features have more samples than max_samples, randomly sample
    if len(features) > max_samples:
        # Create random indices for sampling
        indices = torch.randperm(len(features))[:max_samples]
        features = features[indices]
        labels = labels[indices]
    
    # Prepare data for saving
    embeddings_data = {
        'features': features.cpu().numpy(),
        'labels': labels.cpu().numpy(),
        'step': step
    }
    
    # Save to a .npz file
    save_path = os.path.join(output_dir, f'embeddings_step_{step}.npz')
    np.savez(save_path, **embeddings_data)
    print(f"Embeddings saved to {save_path}")
    return save_path

def visualize_embeddings(features, labels, output_dir, step=0, use_wandb=True, max_samples=100):
    """
    Visualize high-dimensional embeddings using t-SNE, with optional sample limiting.
    
    Args:
    features (torch.Tensor): Feature vectors to visualize
    labels (torch.Tensor): Corresponding labels for the features
    output_dir (str): Directory to save visualization
    step (int, optional): Training step or epoch for logging
    use_wandb (bool, optional): Whether to log to Weights & Biases
    max_samples (int, optional): Maximum number of samples to visualize
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # If features have more samples than max_samples, randomly sample
    if len(features) > max_samples:
        # Create random indices for sampling
        indices = torch.randperm(len(features))[:max_samples]
        features = features[indices]
        labels = labels[indices]
    
    # Move features to CPU and convert to numpy
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_np)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels_np,
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label='Class Labels')
    plt.title(f't-SNE Visualization of Embeddings (Step {step}, {len(features)} samples)')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'embeddings_tsne_step_{step}.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Optionally log to Weights & Biases
    if use_wandb and wandb.run is not None:
        wandb.log({
            f"Embedding Visualization (Step {step}, {len(features)} samples)": wandb.Image(plot_path),
        })
    
    return features_2d



def log_images_to_wandb(missclassified_images: List[dict]):
    """
    Log images to WandB with predicted and original labels.
    
    Args:
        images (torch.Tensor): Batch of images (N, C, H, W).
        predictions (list): Predicted labels for the batch.
        labels (list): Ground-truth labels for the batch.
        num_images (int): Number of images to log.
    """
    images_to_log = []
    for image in missclassified_images:
        img = image["image"].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
        pred_label = image["predicted_label"]
        true_label = image["true_label"]

        # Create WandB image with caption
        images_to_log.append(
            wandb.Image(img, caption=f"Pred: {pred_label}, True: {true_label}")
        )

    # Log to WandB
    wandb.log({"predictions": images_to_log})

def get_missclassified_images_for_logging(data, labels, outputs, num_images_to_log=5):
    misclassified_images = []
    preds = torch.argmax(outputs, dim=1)
    # Find misclassified images and store them
    misclassified_indices = (preds != labels).nonzero(as_tuple=True)[0]
    for idx in misclassified_indices[:num_images_to_log]:  # Limit to a number of images to log
        misclassified_images.append({
            "image": data[idx].cpu(),
            "true_label": labels[idx].item(),
            "predicted_label": preds[idx].item()
        })
    return misclassified_images

def log_confusion_matrix_to_wandb(labels, outputs, class_labels):
    """
    Log confusion matrix to WandB.
    Args:
        confusion_matrix (torch.Tensor): Confusion matrix.
        class_labels (list): List of class labels.
    """
    # Create confusion matrix plot
    confusion_matrix_plot = wandb.plot.confusion_matrix(
        probs=None,
        y_true=labels,
        preds=outputs,
        class_names=class_labels,
    )

    # Log to WandB
    wandb.log({"confusion_matrix": confusion_matrix_plot})

class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images,
                    self.n_last_blocks,
                    return_class_token=True,
                )
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(
        delimiter="  ",
        verbose=distributed.is_main_process(),
    )
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        # outputs is tuple of tuple 4 x (torch.Size([B, 2B, 384]), torch.Size([B, 384]))
        outputs = model(samples.to(device))
        targets = targets.to(device)
        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(
    model,
    dataset,
    batch_size,
    num_workers,
    gather_on_cpu=False,
):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    print(f"sample_count: {sample_count}")
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(
        delimiter="  ",
        verbose=distributed.is_main_process(),
    )
    features, all_labels = None, None
    for samples, (
        index,
        labels_rank,
    ) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(
                sample_count,
                features_rank.shape[-1],
                device=gather_device,
            )
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(
                labels_shape,
                fill_value=-1,
                device=gather_device,
            )
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels



import torch

class PCA:
    def __init__(self, num_components: int):
        """
        Initialize PCA with the number of components to retain.
        """
        self.num_components = num_components
        self.mean = None
        self.components = None

    def fit(self, data: torch.Tensor):
        """
        Compute the principal components from the data.
        :param data: Input data of shape (n_samples, n_features)
        """
        # Compute mean and subtract it
        self.mean = data.mean(dim=0)
        centered_data = data - self.mean

        # Compute covariance matrix
        covariance_matrix = torch.mm(centered_data.T, centered_data) / (data.size(0) - 1)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        self.components = eigenvectors[:, sorted_indices[:self.num_components]]

    def transform(self, data: torch.Tensor):
        """
        Project data onto the principal components.
        :param data: Input data of shape (n_samples, n_features)
        :return: Transformed data of shape (n_samples, num_components)
        """
        centered_data = data - self.mean
        return torch.mm(centered_data, self.components)

    def fit_transform(self, data: torch.Tensor):
        """
        Fit PCA to the data and transform it.
        :param data: Input data of shape (n_samples, n_features)
        :return: Transformed data of shape (n_samples, num_components)
        """
        self.fit(data)
        return self.transform(data)
    
from sklearn.decomposition import IncrementalPCA

class IncrementalPCAWrapper:
    def __init__(self, num_components: int, batch_size: int):
        """
        Initialize IncrementalPCA with the number of components to retain.
        :param num_components: Number of components for PCA
        :param batch_size: Batch size for IncrementalPCA
        """
        self.num_components = num_components
        self.batch_size = batch_size
        self.ipca = IncrementalPCA(n_components=num_components, batch_size=batch_size)

    def partial_fit(self, data: torch.Tensor):
        """
        Incrementally fit the IncrementalPCA model with new data batches.
        :param data: Input data of shape (n_samples, n_features)
        """
        # IncrementalPCA expects numpy arrays, so convert tensor to numpy
        data_np = data.cpu().numpy()
        # Partially fit the model with new data
        self.ipca.partial_fit(data_np)


    def fit(self, data: torch.Tensor):
        """
        Compute the principal components from the data using mini-batches.
        :param data: Input data of shape (n_samples, n_features)
        """
        # IncrementalPCA expects numpy arrays, so convert tensor to numpy
        data_np = data.cpu().numpy()

        # Fit the model incrementally
        self.ipca.fit(data_np)

    def transform(self, data: torch.Tensor):
        """
        Project data onto the principal components using the fitted IncrementalPCA.
        :param data: Input data of shape (n_samples, n_features)
        :return: Transformed data of shape (n_samples, num_components)
        """
        data_np = data.cpu().numpy()

        transformed_data = self.ipca.transform(data_np)

        return torch.tensor(transformed_data, device="cuda")

    def fit_transform(self, data: torch.Tensor):
        """
        Fit IncrementalPCA to the data and transform it.
        :param data: Input data of shape (n_samples, n_features)
        :return: Transformed data of shape (n_samples, num_components)
        """
        self.fit(data)
        return self.transform(data)

