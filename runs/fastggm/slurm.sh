#!/bin/bash
#SBATCH --job-name=fastggm
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

module load singularity-3.0

# Rscript --vanilla run_fastggm.R <data file> <number of cores>
singularity exec im_fastggm.sif Rscript --vanilla run_fastggm.R ${infile} 64

echo
date
