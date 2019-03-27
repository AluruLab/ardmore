#!/bin/bash
#SBATCH --job-name=inferlator
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=1
#SBATCH --time=72:00:00

date
pwd

module load singularity-3.0

# Rscript --vanilla run_inferlator.R <data file> 
singularity exec im_inferlator.sif Rscript /usr/local/bin/inferlator.R config.R

echo
date
