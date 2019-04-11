#!/bin/bash
#SBATCH --job-name=fastggm
#SBATCH --output=./fastggm/logs/log_%J.out
#SBATCH --error=./fastggm/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

nexpts=$(tail -n +2 $datafile | wc -l)
ngenes=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts" 

module load singularity-3.0

mkdir -p fastggm/output

# Rscript --vanilla run_fastggm.R <data file> <output file> <number of cores>
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec $PWD/fastggm/im_fastggm.sif \
	Rscript --vanilla $PWD/fastggm/run_fastggm.R \
	$PWD/$datafile $PWD/fastggm/output/${ngenes}.${nexpts}-${currenttime} \
	64

echo
date
