#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/output/log_%j.out
#SBATCH --time 01:00:00
#SBATCH --partition=dev_accelerated
N_GPUS=1
N_CPUS=19
echo $SLURM_JOB_ID

NUM_COMPONENTS=50
PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/Masterproject-plankton-dinov2 torchrun \
 --standalone --nnodes=1 Masterproject-plankton-dinov2/dinov2/eval/pca.py \
 --num-components=$NUM_COMPONENTS \
 --output-dir='/home/hk-project-p0021769/hgf_grc7525/output/' \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/data/eval/:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/data/eval/:extra=*" \
 --batch-size=64 \
