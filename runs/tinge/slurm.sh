#!/bin/bash
#SBATCH --job-name=tinge
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

module load singularity-3.0

singularity exec im_tinge.sif mpiexec -np 32 /usr/local/bin/tinge-mi -i ${infile} -o tinge.adj -p 0.001 -e 0.1

echo
date
