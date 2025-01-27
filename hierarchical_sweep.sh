#!/bin/sh
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH -e /home/hk-project-p0021769/hgf_rth0695/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_rth0695/output/log_%j.out
#SBATCH --time 01:00:00
#SBATCH --partition=dev_accelerated


export PYTHONPATH=/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2

# Run wandb sweep and capture stdout and stderr separately
wandb agent mp_aqqua/mp_aqqua/iswak5mj