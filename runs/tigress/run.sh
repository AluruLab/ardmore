#!/bin/bash
#SBATCH --job-name=tigress
#SBATCH --output=./tigress/logs/log_%J.out
#SBATCH --error=./tigress/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

nexpts=$(tail -n +2 $datafile | wc -l)
ngenes=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts" 

module load singularity-3.0

mkdir -p tigress/output

# Rscript --vanilla run_tigress.R <data file> <output file> <number of cores>
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec $PWD/tigress/im_tigress.sif \
	Rscript --vanilla $PWD/tigress/run_tigress.R \
	$PWD/$datafile $PWD/tigress/output/${ngenes}.${nexpts}-${currenttime} \
	64

echo
date
