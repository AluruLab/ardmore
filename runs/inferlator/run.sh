#!/bin/bash
#SBATCH --job-name=inferlator
#SBATCH --output=./inferlator/logs/log_%J.out
#SBATCH --error=./inferlator/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

ngenes=$(tail -n +2 $datafile | wc -l)
nexpts=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts"

module load singularity-3.0

mkdir -p inferlator/output
currenttime=$(date "+%Y.%m.%d-%H.%M.%S")

echo "PARS\$input.dir = '$PWD/inferlator/data'
PARS\$exp.mat.file = '$datafile'
PARS\$tf.names.file = '$datafile.genes'
PARS\$save.to.dir = '$PWD/inferlator/output'" >> $PWD/inferlator/config.R

singularity exec \
    $PWD/inferlator/im_inferlator.sif \
    Rscript /usr/local/bin/inferlator.R \
    config.R

echo
date
