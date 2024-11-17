#!/bin/sh

N_GPUS=1
N_CPUS=1
echo $SLURM_JOB_ID
PYTHONPATH=/home/olschulz/projects/Masterproject-plankton-dinov2 torchrun \
	--standalone --nnodes=1 --nproc_per_node=$N_GPUS dinov2/train/train.py --no-resume \
	--config-file dinov2/configs/train/whoi.yaml --run_name=${SLURM_JOB_ID}_${N_GPUS}gpu_pre \
	train.output_dir='/home/olschulz/Documents/output' train.use_torch_compile=true \
	train.dataset_path=LMDBDataset:split=TRAIN:root=/home/olschulz/Documents/lmdbs/random_images:extra=* train.num_workers=$N_CPUS \
	train.augmentations=kornia_cpu student.arch=vit_small crops.use_native_res=false crops.free_shapes=false crops.local_crops_number=4 \
	crops.use_ch_patch_embed=true train.in_chans=3 optim.base_lr=0.0005 crops.use_variable_channels=true \
	student.pretrained_weights=""
