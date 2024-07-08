# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from dinov2.utils.utils import exists, none_or_str

from . import vision_transformer as vits

logger = logging.getLogger("dinov2")


def build_model(
    args,
    only_teacher=False,
    img_size=224,
    free_shapes=None,
    num_loc_crops=8,
    use_ch_patch_embed=False,
    in_chans=3,
):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
            free_shapes=free_shapes,
            num_loc_crops=num_loc_crops,
            use_ch_patch_embed=use_ch_patch_embed,
            in_chans=in_chans,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size,
        free_shapes=none_or_str(cfg.crops.free_shapes),
        num_loc_crops=cfg.crops.local_crops_number,
        use_ch_patch_embed=cfg.crops.use_ch_patch_embed,
        in_chans=cfg.train.in_chans,
    )
