#!/bin/sh
#SBATCH --gres=gpu:full:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=19
#SBATCH --nodes=1
#SBATCH -e /hkfs/work/workspace_haic/scratch/hgf_qvx8970-plankton/output_dir/log_%j.err
#SBATCH -o /hkfs/work/workspace_haic/scratch/hgf_qvx8970-plankton/output_dir/log_%j.out
#SBATCH --time 0-02:00:00
#SBATCH --partition=advanced
#SBATCH --exclude=haicn18[01-03]

N_GPUS=4
N_NODES=1
OMP_NUM_THREADS=2
N_CPUS=19

echo $SLURM_JOB_ID

srun torchrun --standalone \
        --nnodes=1 \
        --nproc_per_node=$N_GPUS \
        --no-resume --ngpus $N_GPUS \
        --num_nodes=1 \
        --config-file dinov2/configs/train/whoi_vitl.yaml \
        --run_name=hai_${SLURM_JOB_ID}_${N_NODES}n_${N_GPUS}gpu_vitl \
        train.num_workers=$N_CPUS \
        train.output_dir=/hkfs/work/workspace_haic/scratch/hgf_qvx8970-plankton/output_dir \
        train.use_torch_compile=true \
        train.batch_size_per_gpu=48 \
        train.dataset_path=LMDBDataset:split=ALL:root=/hkfs/work/workspace_haic/scratch/hgf_qvx8970-plankton/data/lmdb/:extra=* \
        student.pretrained_weights=/hkfs/work/workspace_haic/scratch/hgf_qvx8970-plankton/checkpoints/dinov2_vitl14_pretrain.pth

exit 0