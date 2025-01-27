# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import torch.nn.functional as F
import argparse
from typing import Any, List, Optional, Tuple
from torch.nn import Module
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import dinov2.utils.utils as dinov2_utils
from dinov2.distributed import (
    _restrict_print_to_main_process,
)
from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup
from torchvision.models import ResNet50_Weights, ResNeXt101_32X8D_Weights

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )

    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights, do_eval = False, model_type="dinov2") -> Module:
    if model_type == "dinov2":
        model, _ = build_model_from_cfg(config, only_teacher=True)
        dinov2_utils.load_pretrained_weights(
            model=model,
            pretrained_weights=pretrained_weights,
            checkpoint_key="teacher",
            teacher_student_key="teacher",
            do_eval=do_eval,
        )

    elif model_type == "torchvision":
        base_model = models.resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        # Wrap the model to extract features from the avgpool layer
        model = FeatureExtractorWrapper(base_model, layer_name='avgpool')

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    print("Model loaded from", pretrained_weights)
    print("Model summary:", model)
    model.cuda()
    return model


def setup_and_build_model(args, do_eval: bool = False, model_type="dinov2") -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args, do_eval=do_eval)
    model = build_model_for_eval(config, args.pretrained_weights, do_eval=do_eval, model_type=model_type)
    if do_eval:
        args.output_dir = "/".join(args.pretrained_weights.split("/")[:-1])

    autocast_dtype = get_autocast_dtype(config)
    _restrict_print_to_main_process()
    return model, autocast_dtype


class FeatureExtractorWrapper(torch.nn.Module):
    """Wraps a model to extract features from an intermediate layer."""
    def __init__(self, model: torch.nn.Module, layer_name: str = 'avgpool'):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.features = None
        
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self._get_features_hook)
    
    def _get_features_hook(self, module, input, output):
        self.features = output
    
    def forward(self, x):
        _ = self.model(x)

        features = self.features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        features = F.normalize(features, p=2, dim=1)  # L2 normalize
        return features
