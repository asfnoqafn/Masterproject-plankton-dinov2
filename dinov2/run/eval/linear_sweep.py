# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import torch
import wandb
from dinov2.eval.linear import (
    get_args_parser as get_linear_args_parser,
)
from dinov2.logging import setup_logging_sweep
from dinov2.run.submit import get_args_parser
from dinov2.eval.linear import main as linear_main
import dinov2.distributed as distributed

import yaml
logger = logging.getLogger("dinov2")

def previous_main(args, output_dir):
    try:
        print("Args: ", args)
        updated_args = setup_logging_sweep(args=args, output=output_dir,
        level=logging.INFO,
        do_eval=True,)
        linear_main(updated_args)
    finally:
        if distributed.is_enabled():
            distributed.destroy_process_group()
        if distributed.is_main_process():
            wandb.finish() 
            torch.cuda.empty_cache()
    return 0

def main():
    description = "Submitit launcher for DINOv2 linear evaluation"
    linear_args_parser = get_linear_args_parser(add_help=False)
    parents = [linear_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

     # Ensure clean process group state at start
    if distributed.is_enabled():
        distributed.destroy_process_group()
        torch.cuda.empty_cache()
    
    output_dir = args.output_dir

    with open(os.path.join(args.sweep_config_fpath), "r") as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=lambda: previous_main(args, output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
