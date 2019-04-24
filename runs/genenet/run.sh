#!/bin/bash
#SBATCH --job-name=genenet
#SBATCH --output=./genenet/logs/log_%J.out
#SBATCH --error=./genenet/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

nexpts=$(tail -n +2 $datafile | wc -l)
ngenes=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts" 

module load singularity-3.0

mkdir -p genenet/output

# Rscript --vanilla run_genenet.R <data file> <output file>
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec $PWD/genenet/im_genenet.sif \
	Rscript --vanilla $PWD/genenet/run_genenet.R \
	$PWD/$datafile $PWD/genenet/output/${ngenes}.${nexpts}-${currenttime}

echo
date
