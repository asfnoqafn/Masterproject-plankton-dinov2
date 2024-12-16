#!/bin/sh
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.out
#SBATCH --time 10:00:00
#SBATCH --partition=accelerated
N_GPUS=4
N_CPUS=19
echo $SLURM_JOB_ID

PYTHONPATH=/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2 torchrun \
 --standalone --nnodes=1 dinov2/run/eval/linear.py \
 --config-file dinov2/configs/eval/vits14_pretrain.yaml \
 --output-dir /home/hk-project-p0021769/hgf_rth0695/output/linear \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/eval3:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/eval3/:extra=*" \
 --val-class-mapping-fpath /home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/eval3/VAL_label_map.json \
 --pretrained-weights 'checkpoints/dinov2_vits14_pretrain.pth' \
 --log-missclassified-images True \
 --log-confusion-matrix True \
 --run_name 'linear_eval3' \
