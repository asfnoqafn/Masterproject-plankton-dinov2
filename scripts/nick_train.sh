#!/bin/sh
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.out
#SBATCH --time 12:00:00
#SBATCH --partition=accelerated-h100

BATCH_S=196
N_GPUS=4
N_CPUS=$((32 * $N_GPUS))
echo $SLURM_JOB_ID
export NUMEXPR_MAX_THREADS=128

source ~/.bashrc
micromamba activate dinov2_2

OMP_NUM_THREADS=128 PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2 torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --standalone --nnodes=1 --nproc_per_node=$N_GPUS /home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/train/train.py --no-resume \
 --config-file /home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/configs/train/train_rgb.yaml --run_name=${SLURM_JOB_ID}_${N_GPUS}gpu_pre \
	train.output_dir='/home/hk-project-p0021769/hgf_grc7525/output/' train.use_torch_compile=true \
	train.dataset_path=LMDBDataset:split=ALL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/train/:extra=* train.num_workers=16 \
	train.batch_size_per_gpu=$BATCH_S train.augmentations=kornia_cpu student.arch=vit_small \
	train.in_chans=3

