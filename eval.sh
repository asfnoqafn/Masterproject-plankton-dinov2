#!/bin/bash
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH --mem-per-gpu=45G          # Request 45 GB memory per GPU
#SBATCH --cpus-per-task=8          # Request 8 CPU cores
#SBATCH --chdir=/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2/ # Set working directory
#SBATCH --export=ALL               # Export all environment variables
#SBATCH --error=/home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.err  # Standard error
#SBATCH --output=/home/hk-project-p0021769/hgf_rth0695/output/slurm_output/log_%j.out # Standard output
#SBATCH --time=00:10:00            # Set wall time limit
#SBATCH --partition=dev_accelerated

PYTHONPATH=/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2 torchrun --standalone --nnodes=1 --nproc_per_node=2 /home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2/dinov2/run/eval/knn.py \\
            --config-file /home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2/dinov2/configs/train/whoi_eval.yaml \\
            --pretrained-weights /hkfs/home/project/hk-project-p0021769/hgf_grc7525/checkpoints/dinov2_vits14_pretrain.pth \\
            --output-dir /home/hk-project-p0021769/hgf_rth0695/output/knn \\
            --train-dataset='HDF5Dataset:split=TRAIN:root=/home/hk-project-p0021769/hgf_rth0695/plankton/seanoe_uvplmdb_files:extra=*' \\
            --val-dataset='HDF5Dataset:split=VAL:root=/home/hk-project-p0021769/hgf_rth0695/plankton/seanoe_uvplmdb_files:extra=*'

retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
    exit 100
fi
exit 0
