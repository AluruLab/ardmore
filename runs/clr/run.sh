#!/bin/bash
#SBATCH --job-name=clr
#SBATCH --output=./clr/logs/log_%J.out
#SBATCH --error=./clr/logs/err_%J.err
#SBATCH --ntasks=1
#SBATCH --time=72:00:00

date
pwd

ngenes=$(wc -l < $datafile)
nexpts=$(( $(head -n 1 $datafile | tr -cd , | wc -c)+1 ))
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts"

module load singularity-3.0

mkdir -p clr/output
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
# requires genesXexpts without any headers or row numbers
singularity exec \
    $PWD/clr/im_clr.sif \
    time /usr/local/bin/clr --data $datafile \
    --map $PWD/clr/output/${ngenes}.${nexpts}-$currenttime \
    --bins 10 --spline 3

echo
date
