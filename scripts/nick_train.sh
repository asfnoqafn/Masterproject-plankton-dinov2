#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/repo/output/log_%j.out
#SBATCH --time 00:03:00
#SBATCH --partition=dev_accelerated

BATCH_S=96
N_GPUS=1
N_CPUS=8
echo $SLURM_JOB_ID
OMP_NUM_THREADS=$N_CPUS PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2 torchrun \
	--standalone --nnodes=1 --nproc_per_node=$N_GPUS /home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/train/train.py --no-resume \
	--config-file /home/hk-project-p0021769/hgf_grc7525/repo/Masterproject-plankton-dinov2/dinov2/configs/train/whoi.yaml --run_name=${SLURM_JOB_ID}_${N_GPUS}gpu_pre \
	train.output_dir='/home/hk-project-p0021769/hgf_grc7525/output/' train.use_torch_compile=true \
	train.dataset_path="LMDBDataset:split=ALL:root=/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/bigger/:extra=*" train.num_workers=$N_CPUS \
	train.batch_size_per_gpu=$BATCH_S train.augmentations=kornia_cpu student.arch=vit_small crops.use_native_res=false crops.free_shapes=none \
	crops.local_crops_number=4 crops.use_ch_patch_embed=false train.in_chans=1 optim.base_lr=0.0005 crops.use_variable_channels=false \