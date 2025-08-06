#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -e /home/hk-project-p0021769/hgf_col5747/output/log_%j.err
#SBATCH --output /home/hk-project-p0021769/hgf_col5747/output/log_%j.out
#SBATCH --time 0:50:00
#SBATCH --partition=dev_cpuonly

source ~/.bashrc
micromamba activate dinov2

python /home/hk-project-p0021769/hgf_col5747/Masterproject-plankton-dinov2/notebooks/test_ecotaxa_metadata.py 
# python /home/hk-project-p0021769/hgf_col5747/Masterproject-plankton-dinov2/notebooks/test.py
# python /home/hk-project-p0021769/hgf_col5747/Masterproject-plankton-dinov2/notebooks/data_profiling_presentation.py