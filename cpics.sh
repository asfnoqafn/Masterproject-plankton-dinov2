#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/output/log_%j.out
#SBATCH --time 00:40:00
#SBATCH --partition=dev_cpuonly

PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/Masterproject-plankton-dinov2
torchrun Masterproject-plankton-dinov2/dinov2/data/dataset_creation/save_cpics_to_lmdb.py\
 --dataset_path="/home/hk-project-p0021769/hgf_grc7525/data/with_labels/datasciencebowl/" \
 --lmdb_dir_name="/home/hk-project-p0021769/hgf_grc7525/data/lmdb_without_labels/datasciencebowl/" \
 --min_size="0" --extension=".jpg" --image_folder="test"