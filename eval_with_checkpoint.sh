#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH -e /home/hk-project-p0021769/hgf_twg7490/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_twg7490/output/log_%j.out
#SBATCH --time 00:05:00
#SBATCH --partition=dev_accelerated

N_GPUS=1
N_CPUS=1
echo $SLURM_JOB_ID

if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint>"
    exit 1
fi

CHECKPOINT="$1"
echo "Checkpoint: $CHECKPOINT"

PYTHONPATH=/home/hk-project-p0021769/hgf_twg7490/Masterproject-plankton-dinov2 torchrun \
    --standalone --nnodes=1 dinov2/eval/knn.py \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights=$CHECKPOINT \
    --train-dataset="LMDBDataset:split=TRAIN:root=/hkfs/work/workspace/scratch/hgf_grc7525-nick/eval/:extra=*" \
    --val-dataset="LMDBDataset:split=VAL:root=/hkfs/work/workspace/scratch/hgf_grc7525-nick/eval/:extra=*" \
    --output-dir='/home/hk-project-p0021769/hgf_twg7490/output/'
