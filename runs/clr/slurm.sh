#!/bin/bash
#SBATCH --job-name=clr
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=1
#SBATCH --time=72:00:00

date
pwd

module load singularity-3.0

singularity exec im_clr.sif /usr/local/bin/clr --data ${infile} --map clrnet.output --bins 10 --spline 3

echo
date
