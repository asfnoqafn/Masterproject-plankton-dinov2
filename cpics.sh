#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -e /home/hk-project-p0021769/hgf_auh3910/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_auh3910/output/log_%j.out
#SBATCH --time 00:10:00
#SBATCH --partition=dev_cpuonly

# name of the folder inside the tar
DATASET_NAME=ISIISNet_subset 

# tar file not compressed!
DATASET_TAR=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ISIISNet_subset.tar

# output folder in the workspace
LMDB_FOLDER=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/ISIISNet_subset_lmdb

# the github repo
REPOSITORY_PATH=/home/hk-project-p0021769/hgf_auh3910/repos/Masterproject-plankton-dinov2

source ~/.bashrc
micromamba activate dinov2

export PYTHONPATH=$PYTHONPATH:$REPOSITORY_PATH

tar -C $TMPDIR/ -xf $DATASET_TAR
echo $(ls $TMPDIR)

torchrun $REPOSITORY_PATH/dinov2/data/dataset_creation/convert.py\
 --dataset_path="$TMPDIR/$DATASET_NAME" \
 --lmdb_path="$TMPDIR/result" \
 --min_size="0" --extension=".png" --image_folder="imgs"

rsync -a $TMPDIR/result/ $LMDB_FOLDER
