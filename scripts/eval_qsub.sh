#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_auh3910/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_auh3910/output/log_%j.out
#SBATCH --time 00:20:00
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
 --config-file="dinov2/configs/eval/vits14_reg4_rgb.yaml" \
 --pretrained-weights="/home/hk-project-p0021769/hgf_grc7525/checkpoints/modelgray12h.rank_0.pth" \
 --output-dir="/home/hk-project-p0021769/hgf_auh3910/output/" \
 --knn_output_dir="/home/hk-project-p0021769/hgf_auh3910/output/" \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/eval_meta:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/eval_meta:extra=*" \
 --tensorboard-log-dir="/home/hk-project-p0021769/hgf_auh3910/tensorboard" \
 --save_images \
 train.output_dir='/home/hk-project-p0021769/hgf_auh3910/output/'
