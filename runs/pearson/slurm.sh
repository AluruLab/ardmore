#!/bin/bash
#SBATCH --job-name=pearson
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=1
#SBATCH --time=72:00:00

date
pwd

module load singularity-3.0

# Rscript --vanilla run_pearson.R <data file> 
singularity exec im_pearson.sif Rscript --vanilla run_pearson.R ${infile}

echo
date
