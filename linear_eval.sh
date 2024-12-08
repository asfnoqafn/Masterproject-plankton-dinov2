#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.out
#SBATCH --time 00:08:00
#SBATCH --partition=dev_accelerated
N_GPUS=1
N_CPUS=19
echo $SLURM_JOB_ID

PYTHONPATH=/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2 torchrun \
 --standalone --nnodes=1 dinov2/run/eval/linear.py \
 --config-file dinov2/configs/eval/vits14_pretrain.yaml \
 --output-dir /home/hk-project-p0021769/hgf_rth0695/output/linear \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/plankton:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/plankton/:extra=*" \
 --val-class-mapping-fpath /home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/plankton/-TEST_label_map.json \
 --pretrained-weights 'checkpoints/dinov2_vits14_pretrain.pth' \