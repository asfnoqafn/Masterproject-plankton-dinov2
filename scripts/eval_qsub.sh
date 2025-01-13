#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=1
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.out
#SBATCH --time 00:02:00
#SBATCH --partition=dev_accelerated
N_GPUS=1
N_CPUS=1
echo $SLURM_JOB_ID

PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2 torchrun \
 --standalone --nnodes=1 repo/Masterproject-plankton-dinov2/dinov2/eval/knn.py \
 --config-file repo/Masterproject-plankton-dinov2/dinov2/configs/eval/vits14_pretrain.yaml \
 --pretrained-weights="/home/hk-project-p0021769/hgf_grc7525/checkpoints/model_0011249.rank_0gray.pth" --output-dir \
 /home/hk-project-p0021769/hgf_grc7525/repo/output/ \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooCamNet/:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooCamNet/:extra=*"