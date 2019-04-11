#!/bin/bash
#SBATCH --job-name=irafnet
#SBATCH --output=./irafnet/logs/log_%J.out
#SBATCH --error=./irafnet/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

nexpts=$(tail -n +2 $datafile | wc -l)
ngenes=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts" 

module load singularity-3.0

mkdir -p irafnet/output

# Rscript --vanilla run_irafnet.R <data file> <output file> <number of cores>
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec $PWD/irafnet/im_irafnet.sif \
	Rscript --vanilla $PWD/irafnet/run_irafnet.R \
	$PWD/$datafile $PWD/irafnet/output/${ngenes}.${nexpts}-${currenttime} \

echo
date
