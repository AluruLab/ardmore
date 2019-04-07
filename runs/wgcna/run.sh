#!/bin/bash
#SBATCH --job-name=wgcna
#SBATCH --output=./wgcna/logs/log_%J.out
#SBATCH --error=./wgcna/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

ngenes=$(tail -n +2 $datafile | wc -l)
nexpts=$(head -n 1 $datafile | wc -w)
((nexpts++))

cd $SLURM_SUBMIT_DIR/wgcna
module load singularity-3.0

# Rscript --vanilla run_wgcna.R <data file> <number of cores>
currenttime=$(date "+%Y/%m/%d-%H:%M:%S")
mkdir -p output
singularity exec im_wgcna.sif Rscript --vanilla run_wgcna.R $datafile output/${ngenes}.${nexpts}.${currenttime} 64

echo
date
