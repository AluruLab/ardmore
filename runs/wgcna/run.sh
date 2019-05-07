#!/bin/bash
#SBATCH --job-name=wgcna
#SBATCH --output=./wgcna/logs/log_%J.out
#SBATCH --error=./wgcna/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

nexpts=$(tail -n +2 $datafile | wc -l)
ngenes=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts" 

module load singularity-3.0

mkdir -p wgcna/output

# Rscript --vanilla run_wgcna.R <data file> <output file> <number of cores>
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec $PWD/wgcna/im_wgcna.sif \
	time Rscript --vanilla $PWD/wgcna/run_wgcna.R \
	$PWD/$datafile $PWD/wgcna/output/${ngenes}.${nexpts}-${currenttime} \
	64

echo
date
