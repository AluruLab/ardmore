#!/bin/bash
#SBATCH --job-name=tinge
#SBATCH --output=./tinge/logs/log_%J.out
#SBATCH --error=./tinge/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

ngenes=$(tail -n +2 $datafile | wc -l)
nexpts=$(head -n 1 $datafile | wc -w)
((nexpts--))
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts"

module load singularity-3.0

mkdir -p tinge/output

currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity exec \
    $PWD/tinge/im_tinge.sif \
    mpiexec -np 64 /usr/local/bin/tinge-mi \
    -i $PWD/$datafile \
    -o $PWD/tinge/output/${ngenes}.${nexpts}-${currenttime}
    -p 0.001 -e 0.1

echo
date
