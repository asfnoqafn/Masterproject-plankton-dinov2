#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH -e /home/hk-project-p0021769/hgf_grc7525/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_grc7525/output/log_%j.out
#SBATCH --time 00:20:00
#SBATCH --partition=dev_accelerated

N_GPUS=1
N_CPUS=8
echo $SLURM_JOB_ID
PYTHONPATH=/home/hk-project-p0021769/hgf_grc7525/Masterproject-plankton-dinov2 torchrun \
	--standalone --nnodes=8 --nproc_per_node=$N_GPUS Masterproject-plankton-dinov2/dinov2/run/train/train.py --no-resume --ngpus $N_GPUS \
	--config-file Masterproject-plankton-dinov2/dinov2/configs/train/whoi.yaml --run_name=${SLURM_JOB_ID}_${N_GPUS}gpu_pre \
	train.output_dir='/home/hk-project-p0021769/hgf_grc7525/output' train.use_torch_compile=true \
	train.dataset_path=LMDBDataset:split=ALL:root=/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp/seanoe_uvp_lmdb/:extra=* \
	--train.pretrained_weights='' train.cache_dataset=false train.num_workers=$N_CPUS \
	train.augmentations=kornia_cpu crops.use_native_res=false train.batch_size_per_gpu=16 crops.free_shapes=false \
	crops.local_crops_number=8 student.arch=vit_small crops.use_ch_patch_embed=false train.in_chans=3 crops.use_variable_channels=false
