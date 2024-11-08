#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/output/log_%j.out
#SBATCH --time 00:20:00
#SBATCH --partition=dev_accelerated
N_GPUS=1
N_CPUS=1
echo $SLURM_JOB_ID
PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/Masterproject-plankton-dinov2 torchrun \
	--standalone --nnodes=1 --nproc_per_node=$N_GPUS Masterproject-plankton-dinov2/dinov2/train/train.py --no-resume \
	--config-file Masterproject-plankton-dinov2/dinov2/configs/train/whoi.yaml --run_name=${SLURM_JOB_ID}_${N_GPUS}gpu_pre \
	train.output_dir='/home/hk-project-p0021769/hgf_grc7525/output' train.use_torch_compile=true \
	train.dataset_path=LMDBDataset:split=ALL:root=/home/hk-project-p0021769/hgf_qvx8970/data_lmdb/pisco/:extra=* \
	--train.pretrained_weights='checkpoints/dinov2_vits14_pretrain.pth' train.num_workers=$N_CPUS \
	crops.use_native_res=true crops.free_shapes=true \
	student.arch=vit_small