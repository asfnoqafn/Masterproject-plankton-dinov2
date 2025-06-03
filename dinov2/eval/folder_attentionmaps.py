import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
from dinov2.eval.setup import setup_and_build_model


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    colors = random_colors(N)
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def process_image(img_path, args, model, device, autocast_dtype):
    img = Image.open(img_path)

    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
    ])
    img_tensor = transform(img)
    w, h = img_tensor.shape[1] - img_tensor.shape[1] % args.patch_size, img_tensor.shape[2] - img_tensor.shape[2] % args.patch_size
    img_tensor = img_tensor[:, :w, :h].unsqueeze(0).to(device)

    w_featmap = img_tensor.shape[-2] // 14
    h_featmap = img_tensor.shape[-1] // 14

    # Get all attention maps
    all_attentions = model.get_all_self_attention(img_tensor) # Assuming this function exists in your model

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(args.output_dir2, exist_ok=True)

    attention_maps_list = []
    
    for layer_idx, attentions in enumerate(all_attentions):
        print(f"Layer {layer_idx}: attentions shape: {attentions.shape}, attentions sum: {attentions.sum().item():.4f}")
        nh = attentions.shape[1]
        num_non_patch_tokens = 1 + 4

        cls_to_all = attentions[0, :, 0, :]
        patch_indices = list(range(num_non_patch_tokens, cls_to_all.shape[-1]))
        #print(f"Layer {layer_idx}: CLS token attention shape: {cls_to_all.shape}, Patch indices: {patch_indices}")
        # Compute attention mass
        cls_to_cls = cls_to_all[:, 0]  # (num_heads,)
        cls_to_patches = cls_to_all[:, patch_indices].sum(dim=1)  # (num_heads,)

        print(f"\nLayer {layer_idx}: CLS token attention distribution:")
        for h in range(cls_to_all.shape[0]):
            print(f"Head {h}: CLS→CLS: {cls_to_cls[h].item():.4f}, CLS→Patches: {cls_to_patches[h].item():.4f}")

        print(f"Layer {layer_idx}: Average over heads:")
        print(f"CLS→CLS: {cls_to_cls.mean().item():.4f}")
        print(f"CLS→Patches: {cls_to_patches.mean().item():.4f}")

        cls_to_patch_attn = cls_to_all[:, patch_indices].reshape(nh, h_featmap, w_featmap)
        cls_to_patch_attn = nn.functional.interpolate(cls_to_patch_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
        
        attention_to_non_red = cls_to_cls.mean().item() + cls_to_patches.mean().item()
        # Store attention maps for this layer
        layer_attention_maps = []
        for j in range(nh):
            fname = os.path.join(args.output_dir2, f"{img_name}_layer{layer_idx}_head{j}_#####_{1-cls_to_patches[j]}.png")
            plt.imsave(fname=fname, arr=cls_to_patch_attn[j], format='png')
            #print(f"{fname} saved.")
            layer_attention_maps.append(cls_to_patch_attn[j])
        attention_maps_list.append(layer_attention_maps)
    
    return attention_maps_list # Return the list of all attention maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=14, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path to folder of images to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for attention mask visualization.")
    parser.add_argument('--model_type', default='dinov2', type=str, choices=['dinov2', 'torchvision'],
                        help='Type of model to use for evaluation.')
    parser.add_argument("--config_file", default=None, type=str, help="Path to config file.")
    parser.add_argument("--run_name", default="asd", type=str, help="Name of the run for logging purposes.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Set number of nodes used.")
    parser.add_argument("--output_dir2", default=".", type=str, help="Path to output directory.")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # build model
    model, autocast_dtype = setup_and_build_model(args, do_eval=True , model_type=args.model_type)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # Process all images in folder
    image_folder = args.image_path
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(valid_exts)]

    print(f"Found {len(image_files)} images in {image_folder}")

    for img_path in image_files:
        print(f"Processing {img_path}")
        # Capture the returned list of attention maps if you need them later
        all_attention_maps_for_image = process_image(img_path, args, model, device, autocast_dtype)