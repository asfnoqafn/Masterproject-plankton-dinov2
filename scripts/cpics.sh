#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/output/log_%j.out
#SBATCH --time 01:00:00
#SBATCH --partition=dev_cpuonly

source ~/.bashrc
micromamba activate dinov2_2

PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/Masterproject-plankton-dinov2
torchrun /home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/data/whitepixels.py\
