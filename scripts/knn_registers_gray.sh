#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.out
#SBATCH --time 00:30:00
#SBATCH --partition=accelerated
N_GPUS=1
N_CPUS=19
echo $SLURM_JOB_ID


source ~/.bashrc
micromamba activate dinov2_2

PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2 torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --standalone --nnodes=1 repo/Masterproject-plankton-dinov2/dinov2/eval/knn.py \
 --config-file repo/Masterproject-plankton-dinov2/dinov2/configs/eval/vits14_reg4_rgb.yaml \
 --pretrained-weights="/home/hk-project-p0021769/hgf_grc7525/checkpoints/dinov2_vits14_reg4_pretrain.pth" --output-dir \
 /home/hk-project-p0021769/hgf_grc7525/output/ \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspaces/hkfswork/hgf_grc7525-hiwi/data/TrainTestSplits/ZooCamNet/:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspaces/hkfswork/hgf_grc7525-hiwi/data/TrainTestSplits/ZooCamNet/:extra=*" \
 --output-dir='/home/hk-project-p0021769/hgf_grc7525/output/' \
 --num-workers=$N_CPUS \
 --batch-size=160 \