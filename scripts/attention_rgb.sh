#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=1
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.out
#SBATCH --time 00:03:00
#SBATCH --partition=accelerated

BATCH_S=128
N_GPUS=1
N_CPUS=1
echo $SLURM_JOB_ID

source ~/.bashrc
micromamba activate dinov2_2

PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2 torchrun \
 --rdzv-backend=c10d \
 --rdzv-endpoint=localhost:0 \
 --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
 	/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/eval/rgb_folder_attentionmaps.py \
 	--pretrained_weights="/home/hk-project-p0021769/hgf_grc7525/checkpoints/model_rgb_50h.rank_0.pth" \
	--image_path="/home/hk-project-p0021769/hgf_grc7525/data/ecotaxa/pngs/" \
	--config_file="/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/configs/eval/vits14_reg4_rgb.yaml" \
	--output_dir2="/home/hk-project-p0021769/hgf_grc7525/attention_visualizations/zooscann_rgb_reg/" \