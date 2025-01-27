#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH -e /home/hk-project-p0021769/hgf_rth0695/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_rth0695/output/log_%j.out
#SBATCH --time 01:00:00
#SBATCH --partition=cpuonly

PYTHONPATH=/home/hk-project-p0021769/hgf_rth0695/Masterproject-plankton-dinov2
torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
    dinov2/data/dataset_creation/train_test_split.py \
 --dataset_path="/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/ZooScanNet" \
 --lmdb_dir_name="/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/TrainTestSplits/ZooScanNet" \
 --min_size="0" \