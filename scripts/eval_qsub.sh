#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/output/log_%j.out
#SBATCH --time 04:00:00
#SBATCH --partition=dev_accelerated
N_GPUS=1
N_CPUS=19
echo $SLURM_JOB_ID

PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2 torchrun \
 --standalone --nnodes=1 repo/Masterproject-plankton-dinov2/dinov2/eval/knn.py \
 --config-file repo/Masterproject-plankton-dinov2/dinov2/configs/eval/vits14_pretrain.yaml \
 --pretrained-weights="/home/hk-project-p0021769/hgf_grc7525/checkpoints/dinov2_vits14_pretrain.pth" --output-dir \
 /home/hk-project-p0021769/hgf_grc7525/output/ \
<<<<<<< HEAD:eval_qsub.sh
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/eval/:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/eval/:extra=*" \
 --output-dir='/home/hk-project-p0021769/hgf_grc7525/output/'
=======
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/eval3/:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/eval3/:extra=*" \
 --output-dir='/home/hk-project-p0021769/hgf_grc7525/output/'
>>>>>>> merge:scripts/eval_qsub.sh
