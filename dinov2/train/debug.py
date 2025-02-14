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
            
            print(crops.shape)

            grid = vutils.make_grid(crops, nrow=8, padding=2, normalize=True)
            print(grid.shape)
            plt.figure(figsize=(20, 20))
            plt.imshow(grid.cpu().to(torch.float32).numpy().transpose(1, 2, 0))
            plt.axis('off')
            plt.savefig(os.path.join(debug_dir, f'{prefix}.png'))
            plt.close()
        
        # Save global crops
        print('saving global crops')
        #save_patches(data['collated_global_crops'], 'global_crops')
        if 'collated_local_crops' in data:
            print('saving local crops')
            save_patches(data['collated_local_crops'], 'local_crops')
        
        # Save mask visualizations if available
        # if 'collated_masks' in data:
        #     mask_dir = os.path.join(debug_dir, 'masks')
        #     os.makedirs(mask_dir, exist_ok=True)
        #     masks = data['collated_masks']
        #     for i, mask in enumerate(masks):
        #         plt.figure(figsize=(10, 10))
        #         plt.imshow(mask.cpu().numpy(), cmap='gray')
        #         plt.axis('off')
        #         plt.savefig(os.path.join(mask_dir, f'mask_{i}.png'))
        #         plt.close()
        
        return True, nan_losses
    
    return False, []