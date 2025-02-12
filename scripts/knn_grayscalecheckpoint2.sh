#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=1
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.out
#SBATCH --time 00:03:00
#SBATCH --partition=dev_accelerated-h100
N_GPUS=1
N_CPUS=1
echo $SLURM_JOB_ID

PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2 torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --standalone --nnodes=1 repo/Masterproject-plankton-dinov2/dinov2/eval/knn.py \
 --config-file repo/Masterproject-plankton-dinov2/dinov2/configs/eval/vits14_pretrain_grayscale2.yaml \
 --pretrained-weights="/home/hk-project-p0021769/hgf_grc7525/checkpoints/dinov2_vits14_reg4_pretrain.pth" --output-dir \
 /home/hk-project-p0021769/hgf_grc7525/output/ \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/plankton/:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/plankton/:extra=*" \
 --output-dir='/home/hk-project-p0021769/hgf_grc7525/output/'