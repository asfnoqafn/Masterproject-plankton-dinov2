#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH -e /home/hk-project-p0021769/hgf_rth0695/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_rth0695/output/log_%j.out
#SBATCH --time 01:00:00
#SBATCH --partition=dev_accelerated
BATCH_S=128
N_GPUS=1
N_CPUS=$((32 * $N_GPUS))
echo $SLURM_JOB_ID
export NUMEXPR_MAX_THREADS=128

OMP_NUM_THREADS=64 PYTHONPATH=/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2 torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --nproc_per_node=$N_GPUS \
 --standalone --nnodes=1 dinov2/run/eval/linear.py \
 --config-file dinov2/configs/train/vits14_pretrain_grayscale2.yaml \
 --output-dir /home/hk-project-p0021769/hgf_rth0695/output/linear/ZooScanNet \
 --train_dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooCamNet:extra=*" \
 --val_dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooCamNet:extra=*" \
 --pretrained-weights 'checkpoints/modelgray12h.rank_0.pth' \
 --run_name 'linear_eval_vits14_ZooScanNet' \
 --loss_function="cross_entropy" \
 --gray_scale True \