#!/bin/bash
#SBATCH --job-name=pearson
#SBATCH --output=./pearson/logs/log_%J.out
#SBATCH --error=./pearson/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

nexpts=$(tail -n +2 $datafile | wc -l)
ngenes=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts" 

module load singularity-3.0

mkdir -p pearson/output

# Rscript --vanilla run_pearson.R <data file> <output file> <number of cores>
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec $PWD/pearson/im_pearson.sif \
	Rscript --vanilla $PWD/pearson/run_pearson.R \
	$PWD/$datafile $PWD/pearson/output/${ngenes}.${nexpts}-${currenttime} \

echo
date
