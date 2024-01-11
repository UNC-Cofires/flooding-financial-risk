#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=140g
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=0-6
export OMP_NUM_THREADS=$SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export OPENBLAS_NUM_THREADS=$SLURM_NTASKS
export BLIS_NUM_THREADS=$SLURM_NTASKS
export VECLIB_MAXIMUM_THREADS=$SLURM_NTASKS
export NUMEXPR_NUM_THREADS=$SLURM_NTASKS
cd /proj/characklab/flooddata/NC/multiple_events/analysis
module purge
module load anaconda/2023.03
conda activate /proj/characklab/flooddata/NC/conda_environments/envs/flood_v1
python3.11 estimate_damages.py
