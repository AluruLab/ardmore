#!/bin/bash
#SBATCH --job-name=aracne
#SBATCH --output=log%J.out
#SBATCH --error=err%J.err
#SBATCH --ntasks=1
#SBATCH --time=72:00:00

date
pwd

module load singularity-3.0

singularity exec im_aracne.sif java -Xmx5G -jar /usr/local/bin/aracne.jar -e ${infile} -o . --tfs ${genes} --pvalue 1E-8 --seed 1 --calculateThreshold
singularity exec im_aracne.sif java -Xmx5G -jar /usr/local/bin/aracne.jar -e ${infile} -o . --tfs ${genes} --pvalue 1E-8 --seed 1

echo
date
