# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note the original is here: https://github.com/facebookresearch/dino/blob/main/visualize_attention.py

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as pth_transforms

from dinov2.eval.setup import build_model_for_eval
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.utils.config import setup


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name for the wandb log",
        default="viz_attn_run",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        help="Image for the visualization",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Set number of nodes used."
    )
    return parser


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = setup(args, do_eval=True)
    model = build_model_for_eval(config, args.pretrained_weights)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    original_image = Image.open(args.img_path).convert("RGB")

    width, height = original_image.size
    aspect_ratio = width / height if width > height else height / width
    if width > height:
        new_width = config.visualization.max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = config.visualization.max_size
        new_width = int(new_height / aspect_ratio)
    image_size = (new_height, new_width)

    MEAN = [0.68622917, 0.68622917, 0.68622917]
    STD = [0.10176649, 0.10176649, 0.10176649]

    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(MEAN, STD),
        ]
    )
    img = transform(original_image)

    patch_size = config.student.patch_size

    # Make the image divisible by the patch size
    width, height = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )
    img = img[:, :width, :height].unsqueeze(0)

    width_featmap = img.shape[-2] // patch_size
    height_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_self_attention(img.to(device))

    num_heads = attentions.shape[1]

    # Keep only the output patch attention for every patch
    attentions = attentions[0, :, 0, 1:].reshape(num_heads, -1)
    attentions = (
        attentions.reshape(num_heads, width_featmap, height_featmap).cpu().numpy()
    )

    attentions = np.append(
        attentions, np.mean(attentions, axis=0)[np.newaxis, :, :], axis=0
    )

    print(attentions.shape)
    os.makedirs(os.path.join(args.output_dir, args.run_name, "figs"), exist_ok=True)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    titles = [
        "Image",
        "Head 0",
        "Head 1",
        "Head 2",
        "Head Average",
        "Head 3",
        "Head 4",
        "Head 5",
    ]

    images = [
        original_image,
        attentions[0],
        attentions[1],
        attentions[2],
        attentions[-1],
        attentions[3],
        attentions[4],
        attentions[5],
    ]

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()
    save_path = os.path.join(
        args.output_dir, args.run_name, "figs", "attention" + ".png"
    )
    print("saving to ", save_path)
    plt.savefig(save_path)


if __name__ == "__main__":
    description = "DINOv2 attention visualization"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
