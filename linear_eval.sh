#!/bin/sh
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.out
#SBATCH --time 01:00:00
#SBATCH --partition=dev_accelerated
N_GPUS=4
N_CPUS=19
echo $SLURM_JOB_ID
export TORCH_USE_CUDA_DSA=1

PYTHONPATH=/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2 torchrun \
 --standalone --nnodes=1 dinov2/run/eval/linear.py \
 --config-file dinov2/configs/eval/vits14_pretrain.yaml \
 --output-dir /home/hk-project-p0021769/hgf_rth0695/output/linear/ZooCamNet \
 --train-dataset="LMDBDataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooCamNet:extra=*" \
 --val-dataset="LMDBDataset:split=VAL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooCamNet:extra=*" \
 --val-class-mapping-fpath="/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooCamNet/VAL_label_map.json" \
 --pretrained-weights 'checkpoints/dinov2_vits14_pretrain.pth' \
 --log-missclassified-images True \
 --log-confusion-matrix True \
 --run_name 'linear_eval_ZooCamNet'
#  --hierarchy-file-path="/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2/hierarchy_zoo_cam.json" \
#  --loss-function="custom_hierarchical" \