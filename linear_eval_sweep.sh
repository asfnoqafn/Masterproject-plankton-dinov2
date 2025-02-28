#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH -e /home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.out
#SBATCH --time 00:10:00
#SBATCH --partition=accelerated
BATCH_S=128
N_GPUS=1
N_CPUS=$((32 * $N_GPUS))
echo $SLURM_JOB_ID
export NUMEXPR_MAX_THREADS=128

source ~/.bashrc
micromamba activate dinov2

OMP_NUM_THREADS=64 PYTHONPATH=/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2 torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --nproc_per_node=$N_GPUS \
 --standalone --nnodes=1 dinov2/run/eval/linear_sweep.py \
 --config-file dinov2/configs/eval/vits14_pretrain.yaml \
 --output-dir /home/hk-project-p0021769/hgf_rth0695/output/linear/ZooScanNet \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooScanNet:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooScanNet:extra=*" \
 --val-class-mapping-fpath="/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooScanNet/VAL_label_map.json" \
 --pretrained-weights 'checkpoints/dinov2_vits14_pretrain.pth' \
 --log-missclassified-images True \
 --log-confusion-matrix True \
 --run_name 'linear_eval_vits14_ZooScanNet' \
 --hierarchy-file-path="/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2/hierarchy_zoo_scan.json" \
 --loss-function="custom_hierarchical_combined" \
 --sweep-config-fpath="/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2/sweep_config.yaml" \
 --save-checkpoint False \