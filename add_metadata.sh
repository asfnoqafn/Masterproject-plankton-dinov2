#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -e /home/hk-project-p0021769/hgf_auh3910/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_auh3910/output/log_%j.out
#SBATCH --time 00:10:00
#SBATCH --partition=dev_cpuonly


# output folder in the workspace
LMDB_FOLDER=/home/hk-project-p0021769/hgf_auh3910/workspace/data/lmdb_with_labels/ISIISNet

# the github repo
REPOSITORY_PATH=/home/hk-project-p0021769/hgf_auh3910/repos/Masterproject-plankton-dinov2

# 43.69N, 43.29S, 7.79E, 7.32W
METADATA='{name: "ISIISNet", north: 43.69, south: 43.29, east: 7.79, west: 7.32}'

source ~/.bashrc
micromamba activate dinov2

export PYTHONPATH=$PYTHONPATH:$REPOSITORY_PATH



torchrun $REPOSITORY_PATH/dinov2/data/dataset_creation/append_metadata_lmdb.py\
 --lmdb_path="$LMDB_FOLDER" \
 --metadata="$METADATA" \


