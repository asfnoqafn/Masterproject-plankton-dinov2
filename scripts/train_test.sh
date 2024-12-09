#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/output/log_%j.out
#SBATCH --time 01:40:00
#SBATCH --partition=cpuonly

PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/Masterproject-plankton-dinov2
torchrun Masterproject-plankton-dinov2/dinov2/data/dataset_creation/train_test_split.py\
 --dataset_path="/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/lmdb_with_labels/" \
 --lmdb_dir_name="/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/eval2/" \
 --min_size="0"