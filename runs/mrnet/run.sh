#!/bin/bash
#SBATCH --job-name=mrnet
#SBATCH --output=./mrnet/logs/log_%J.out
#SBATCH --error=./mrnet/logs/err_%J.err
#SBATCH --ntasks=1
#SBATCH --time=72:00:00

date
pwd

nexpts=$(tail -n +2 $datafile | wc -l)
ngenes=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts" 

module load singularity-3.0

mkdir -p mrnet/output

# Rscript --vanilla run_mrnet.R <data file> <output file>
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec $PWD/mrnet/im_mrnet.sif \
	time Rscript --vanilla $PWD/mrnet/run_mrnet.R \
	$PWD/$datafile $PWD/mrnet/output/${ngenes}.${nexpts}-${currenttime}

echo
date
