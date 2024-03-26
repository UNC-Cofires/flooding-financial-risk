#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=120g
#SBATCH -t 4-00:00:00
#SBATCH --job-name=loan_survival
#SBATCH --mail-type=all
#SBATCH --mail-user=kieranf@email.unc.edu

cd /proj/characklab/flooddata/NC/multiple_events/analysis
module purge
module load r/4.2.1
Rscript loan_survival_analysis.R