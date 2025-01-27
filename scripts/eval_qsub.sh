#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_auh3910/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_auh3910/output/log_%j.out
#SBATCH --time 00:10:00
#SBATCH --partition=accelerated
N_GPUS=1
N_CPUS=19
echo $SLURM_JOB_ID


# the github repo
REPOSITORY_PATH=/home/hk-project-p0021769/hgf_auh3910/repos/Masterproject-plankton-dinov2

source ~/.bashrc
micromamba activate dinov2

export PYTHONPATH=$PYTHONPATH:$REPOSITORY_PATH

torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --standalone \
 --nnodes=1 \
 dinov2/eval/knn.py \
 --config-file dinov2/configs/eval/vits14_pretrain.yaml \
 --pretrained-weights="/home/hk-project-p0021769/hgf_grc7525/checkpoints/dinov2_vits14_pretrain.pth" \
 --output-dir="/home/hk-project-p0021769/hgf_auh3910/output/" \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooScanNet:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooScanNet:extra=*"
