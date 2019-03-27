#!/bin/bash
#SBATCH --job-name=genie3
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

module load singularity-3.0

# Rscript --vanilla run_genie3.R <data file> 
singularity exec im_genie3.sif Rscript --vanilla run_genie3.R ${infile} 64

echo
date
