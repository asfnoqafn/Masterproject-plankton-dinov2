#!/bin/sh
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=19
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.out
#SBATCH --time 00:03:00
#SBATCH --partition=accelerated

BATCH_S=128
N_GPUS=1
N_CPUS=19
echo $SLURM_JOB_ID

source ~/.bashrc
micromamba activate dinov2_2

OMP_NUM_THREADS=20 PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2 torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --standalone --nnodes=1 --nproc_per_node=$N_GPUS /home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/train/train.py --no-resume \
 --config-file /home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/configs/train/train_rgb.yaml --run_name=${SLURM_JOB_ID}_${N_GPUS}gpu_pre \
	student.pretrained_weights=" "\
	train.output_dir='/home/hk-project-p0021769/hgf_grc7525/output/'\
	train.use_torch_compile=true \
	train.dataset_path=LMDBDataset:split=ALL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/plankton/:extra=* \
	train.num_workers=$N_CPUS \
	train.batch_size_per_gpu=$BATCH_S \
	crops.use_ch_patch_embed=false train.in_chans=3 

