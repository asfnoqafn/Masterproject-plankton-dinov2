#$ -l gpu=2
#$ -l cuda_memory=40G
#$ -pe smp 8
#$ -cwd
#$ -V
#$ -e /home/jluesch/output_dir/log_$JOB_ID.err
#$ -o /home/jluesch/output_dir/log_$JOB_ID.out
#$ -l h_rt=1:00:00
#$ -A kainmueller
#$ -l h=maxg09


BATCH_S=16
N_GPUS=2
N_CPUS=8
echo $JOB_ID
OMP_NUM_THREADS=$N_CPUS PYTHONPATH=/fast/AG_Kainmueller/jluesch/plankton-dinov2 torchrun --standalone --nnodes=1 \
        --nproc_per_node=$N_GPUS dinov2/run/train/train.py --no-resume \
        --ngpus $N_GPUS --config-file dinov2/configs/train/whoi.yaml \
        --run_name=${JOB_ID}_${N_GPUS}gpu_whoi_bs${BATCH_S}_chEmb \
        train.output_dir=/fast/AG_Kainmueller/plankton/output_dir \
        train.use_torch_compile=true \
        train.dataset_path=LMDBDataset:split=ALL:root=/fast/AG_Kainmueller/plankton/data/WHOI/preprocessed/preprocessed_nat_lmdb:extra=* \
        train.batch_size_per_gpu=$BATCH_S train.num_workers=$N_CPUS train.augmentations=kornia_cpu \
        student.arch=vit_small crops.use_native_res=false crops.free_shapes=false crops.local_crops_number=4 \
        crops.use_ch_patch_embed=true train.in_chans=1 optim.base_lr=0.0005 crops.use_variable_channels=true \
        student.pretrained_weights="" \