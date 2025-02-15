import torch
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
import numpy as np

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