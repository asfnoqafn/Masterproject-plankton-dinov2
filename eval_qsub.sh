#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/output/log_%j.out
#SBATCH --time 00:20:00
#SBATCH --partition=dev_accelerated-h100
N_GPUS=1
N_CPUS=1
echo $SLURM_JOB_ID

PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/Masterproject-plankton-dinov2 torchrun \
/home/hk-project-p0021769/hgf_grc7525/Masterproject-plankton-dinov2/dinov2/eval/knn.py --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/out2:extra=*" \
--val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/out2/:extra=*" \
--config-file "/home/hk-project-p0021769/hgf_grc7525/Masterproject-plankton-dinov2/dinov2/configs/eval/vits14_pretrain.yaml" \
--pretrained-weights '/home/hk-project-p0021769/hgf_grc7525/checkpoints/dinov2_vits14_pretrain.pth' --output-dir \
/home/hk-project-p0021769/hgf_grc7525/output/