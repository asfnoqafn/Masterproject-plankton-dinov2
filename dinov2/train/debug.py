import torch
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
import dinov2.distributed as distributed


def debug_nan_losses(loss_dict, data, cfg, iteration, output_dir):
    nan_detected = False
    nan_losses = []
    
    for loss_name, loss_value in loss_dict.items():
        if torch.isnan(loss_value):
            nan_detected = True
            nan_losses.append(loss_name)
    
    if nan_detected:
        debug_dir = os.path.join(output_dir, f'nan_debug_iter_{iteration}')
        os.makedirs(debug_dir, exist_ok=True)
        
        with open(os.path.join(debug_dir, 'nan_losses.txt'), 'w') as f:
            f.write(f"NaN detected in losses: {', '.join(nan_losses)}\n")
            f.write(f"All loss values:\n")
            for name, value in loss_dict.items():
                f.write(f"{name}: {value.item()}\n")
        

        def save_patches(crops, prefix):
            if crops is None:
                return
            
            debug_dir = os.path.join(output_dir, f'nan_debug_iter_{iteration}')
            
            num_images = crops.shape[0]  # Total number of images
            batch_size = 64  # Define reasonable batch size per plot
            num_batches = (num_images + batch_size - 1) // batch_size  # Number of batches

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_images)
                batch = crops[start_idx:end_idx]  # Extract batch
                
                grid = vutils.make_grid(batch, nrow=8, padding=2, normalize=True)
                
                plt.figure(figsize=(20, 20))
                plt.imshow(grid.cpu().to(torch.float32).numpy().transpose(1, 2, 0))
                plt.axis('off')
                plt.savefig(os.path.join(debug_dir, f'{prefix}_batch_{i}.png'))
                plt.close()
        

        if 'collated_local_crops' in data:
            print('saving local crops')
            save_patches(data['collated_local_crops'], 'local_crops')
        

        
        return True, nan_losses
    
    return False, []



def compute_image_entropy(image_patches, num_bins=256):
    batch_size, channels, height, width = image_patches.shape
    flattened_patches = image_patches.view(batch_size, -1).cpu()

    histograms = torch.stack([
        torch.histogram(flattened_patches[b], bins=num_bins, range=(0, 1), density=True)[0]
        for b in range(batch_size)
    ]).view(batch_size, num_bins)

    probs = histograms / histograms.sum(dim=1, keepdim=True)  # Shape: (batch_size, channels, num_bins)

    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)  # Shape: (batch_size, channels)

    return entropy

def softmax_entropy(embeddings):
    probs = torch.softmax(embeddings, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    return entropy


def visualize_entropy_distribution(entropy_values, title="Embedding Entropy Distribution"):
    """
    Create a histogram of entropy values
    Args:
        entropy_values: torch.Tensor or numpy array of entropy values
        title: string for plot title
    """
    if torch.is_tensor(entropy_values):
        entropy_values = entropy_values.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(entropy_values, bins=100, density=False, alpha=0.7)
    plt.xlabel("Entropy")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def log_data_entropy(data, iteration):
    if distributed.is_main_process():
        entropy_dict = {}
        
        with torch.no_grad():

            local_crops = data["collated_local_crops"].float()
            lc_reshaped = local_crops.reshape(local_crops.shape[0], -1)
            per_patch_entropy = softmax_entropy(lc_reshaped)
            current_stats = {}
            
            fig = visualize_entropy_distribution(
                per_patch_entropy,
                title=f"Local Crops Per-Patch Entropy (Iteration {iteration})"
            )
            current_stats["local_crops_entropy_distribution"] = wandb.Image(fig)
            plt.close(fig)

            entropy_dict.update(current_stats)
        
        print("Entropy dict:", entropy_dict)
        wandb.log(entropy_dict, step=iteration)

def log_entropy(data,cls,iteration):
    if distributed.is_main_process():
        #print("Logging entropy")
        with torch.no_grad():
            #### cls
            #print("CLS shape:", cls.shape)
            per_cls_entropy = softmax_entropy(cls)
            current_stats = {}
            current_stats["cls_entropy_min"] = per_cls_entropy.min()
            current_stats["cls_entropy_max"] = per_cls_entropy.max()
            current_stats["cls_entropy_mean"] = per_cls_entropy.mean()
            fig = visualize_entropy_distribution(
                per_cls_entropy,
                title=f"CLS Entropy Distribution (Iteration {iteration})"
            )
            current_stats["CLS Entropy Distribution"] = wandb.Image(fig)
            plt.close(fig)

            local_crops = data.float()
          
            per_patch_entropy = compute_image_entropy(local_crops)
            
            fig = visualize_entropy_distribution(
                per_patch_entropy,
                title=f"Local Crops Per-Patch Entropy (Iteration {iteration})"
            )
            current_stats["local_crops_entropy_distribution"] = wandb.Image(fig)
            plt.close(fig)

        
            print("Entropy dict:", current_stats)
            wandb.log(current_stats, step=iteration)


def log_cls_similarities(cls_tokens, iteration):
    if distributed.is_main_process():
        with torch.no_grad():
            number_nan = torch.isnan(cls_tokens).sum()
            # wierd hacky way to calculate similarity
            normalized_embeddings = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)
            similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
            
            # Exclude self
            mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
            mean_similarity = similarities[mask].mean()
            print("Mean similarity:", mean_similarity)
            wandb.log({"cls_mean_similarity": mean_similarity, "number_nan_cls": number_nan}, step=iteration)

def log_cls_similarities2(cls_tokens, iteration):
    if distributed.is_main_process():
        with torch.no_grad():
            # Normalize embeddings
            normalized_embeddings = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)
            
            # Compute pairwise cosine similarity
            similarity_matrix = torch.nn.functional.cosine_similarity(
                normalized_embeddings.unsqueeze(1), normalized_embeddings.unsqueeze(0), dim=2
            )
            
            # Exclude self-similarities
            mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
            mean_similarity = similarity_matrix[mask].mean()
            print("Mean similarity:", mean_similarity)
            wandb.log({"cls_mean_similarity": mean_similarity}, step=iteration)