#!/bin/bash
#SBATCH --job-name=genenet
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

module load singularity-3.0

# Rscript --vanilla run_genenet.R <data file> 
singularity exec im_genenet.sif Rscript --vanilla run_genenet.R ${infile}

echo
date
