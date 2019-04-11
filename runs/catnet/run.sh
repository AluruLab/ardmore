#!/bin/bash
#SBATCH --job-name=catnet
#SBATCH --output=./catnet/logs/log_%J.out
#SBATCH --error=./catnet/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

nexpts=$(tail -n +2 $datafile | wc -l)
ngenes=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts" 

module load singularity-3.0

mkdir -p catnet/output

# Rscript --vanilla run_catnet.R <data file> <output file> <number of cores>
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec $PWD/catnet/im_catnet.sif \
	Rscript --vanilla $PWD/catnet/run_catnet.R \
	$PWD/$datafile $PWD/catnet/output/${ngenes}.${nexpts}-${currenttime} \
	64

echo
date
