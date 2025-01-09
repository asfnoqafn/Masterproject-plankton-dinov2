#!/bin/sh
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=32
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.out
#SBATCH --time 00:45:00
#SBATCH --partition=accelerated-h100

BATCH_S=128
N_GPUS=4
N_CPUS=$((32 * $N_GPUS))
echo $SLURM_JOB_ID
export NUMEXPR_MAX_THREADS=128

source ~/.bashrc
micromamba activate dinov2

OMP_NUM_THREADS=128 PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2 torchrun \
	--standalone --nnodes=1 --nproc_per_node=$N_GPUS /home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/train/train.py --no-resume \
	--config-file /home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/configs/train/whoi.yaml --run_name=${SLURM_JOB_ID}_${N_GPUS}gpu_pre \
	train.output_dir='/home/hk-project-p0021769/hgf_grc7525/output/' train.use_torch_compile=true \
	train.dataset_path=LMDBDataset:split=ALL:root=/hkfs/work/workspace/scratch/hgf_grc7525-nick/train/nontest/:extra=* train.num_workers=$N_CPUS \
	train.batch_size_per_gpu=$BATCH_S train.augmentations=kornia_cpu student.arch=vit_small crops.use_native_res=false crops.free_shapes=none \
	crops.use_ch_patch_embed=false train.in_chans=1 optim.base_lr=0.001 crops.use_variable_channels=false