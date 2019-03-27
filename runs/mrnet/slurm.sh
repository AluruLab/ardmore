#!/bin/bash
#SBATCH --job-name=mrnet
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=1
#SBATCH --time=72:00:00

date
pwd

module load singularity-3.0

# Rscript --vanilla run_mrnet.R <data file> 
singularity exec im_mrnet.sif Rscript --vanilla run_mrnet.R ${infile}

echo
date
