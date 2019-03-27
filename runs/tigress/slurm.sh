#!/bin/bash
#SBATCH --job-name=tigress
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

module load singularity-3.0

# Rscript --vanilla run_tigress.R <data file> 
singularity exec im_tigress.sif Rscript --vanilla run_tigress.R ${infile} 64

echo
date
