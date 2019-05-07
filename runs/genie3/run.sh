#!/bin/bash
#SBATCH --job-name=genie3
#SBATCH --output=./genie3/logs/log_%J.out
#SBATCH --error=./genie3/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

ngenes=$(tail -n +2 $datafile | wc -l)
nexpts=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts"

module load singularity-3.0

mkdir -p genie3/output

# Rscript --vanilla run_genie3.R <data file> <output file> <ncores>
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec \
    $PWD/genie3/im_genie3.sif \
    time Rscript --vanilla $PWD/genie3/run_genie3.R \
    $PWD/$datafile $PWD/genie3/output/${ngenes}.${nexpts}-$currenttime \
    64

echo
date
