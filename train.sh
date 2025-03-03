#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_rth0695/repo/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_rth0695/repo/output/log_%j.out
#SBATCH --time 00:20:00
#SBATCH --partition=dev_accelerated
N_GPUS=1
N_CPUS=19
echo $SLURM_JOB_ID

PYTHONPATH=/home/hk-project-p0021769/hgf_rth0695/repo/Masterproject-plankton-dinov2 torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --standalone --nnodes=1 repo/Masterproject-plankton-dinov2/dinov2/eval/knn.py \
 --config-file repo/Masterproject-plankton-dinov2/dinov2/configs/eval/vits14_pretrain_grayscale2.yaml \
 --pretrained-weights="/home/hk-project-p0021769/hgf_rth0695/checkpoints/modelgray12h.rank_0.pth" --output-dir \
 /home/hk-project-p0021769/hgf_rth0695/repo/output/ \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_rth0695/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooScanNet/:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_rth0695/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooScanNet/:extra=*" \
 --output-dir='/home/hk-project-p0021769/hgf_rth0695/output/' \
 --save_images