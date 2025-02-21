# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import math
import os
from datetime import datetime

from omegaconf import OmegaConf

import dinov2.distributed as distributed
from dinov2.configs import dinov2_default_config
from dinov2.logging import setup_logging
from dinov2.utils import utils

logger = logging.getLogger("dinov2")


def apply_scaling_rules_to_cfg(cfg):  # to fix
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    # args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def default_setup(args, output_dir, do_eval: bool = False):
    distributed.enable(overwrite=True, num_nodes=args.num_nodes)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    # if distributed.is_main_process():
    #     setup_logging(
    #         args=args,
    #         output=output_dir,
    #         level=logging.INFO,
    #         do_eval=do_eval,
    #     )

    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args, do_eval: bool = False):
    """
    Create configs and perform basic setups.
    """

    args.run_name = args.run_name + f"_{datetime.now().strftime('%d%m%Y_%H%M%S')}"
    cfg = get_cfg_from_args(args)
    args.run_name = args.run_name + "_" + cfg.student.arch
    print("args.run_name ", args.run_name)

    if len(cfg.train.output_dir) > 4:
        cfg.train.output_dir = os.path.join(cfg.train.output_dir, args.run_name)
    else:
        cfg.train.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    default_setup(
        args,
        output_dir=cfg.train.output_dir,
        do_eval=do_eval,
    )
    apply_scaling_rules_to_cfg(cfg)
    if distributed.is_main_process():
        write_config(cfg, cfg.train.output_dir)
    return cfg
