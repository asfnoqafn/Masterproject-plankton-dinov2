#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=38
#SBATCH -e /home/hk-project-p0021769/hgf_auh3910/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_auh3910/output/log_%j.out
#SBATCH --time 00:10:00
#SBATCH --partition=accelerated-h100
N_GPUS=1
N_CPUS=38
echo $SLURM_JOB_ID

# the github repo
REPOSITORY_PATH=/home/hk-project-p0021769/hgf_auh3910/repos/Masterproject-plankton-dinov2

source ~/.bashrc
micromamba activate dinov2

export PYTHONPATH=$PYTHONPATH:$REPOSITORY_PATH


export NUMEXPR_MAX_THREADS=128
OMP_NUM_THREADS=19

torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --standalone \
 --nnodes=1 \
 --nproc_per_node=$N_GPUS \
  scripts/augmentation_test.py