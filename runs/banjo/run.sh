#!/bin/bash
#SBATCH --job-name=banjo
#SBATCH --output=./banjo/logs/log_%J.out
#SBATCH --error=./banjo/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

nexpts=$(wc -l)
ngenes=$(head -n 1 $datafile | wc -w)
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts"

module load singularity-3.0

mkdir -p banjo/output

singularity exec $PWD/banjo/im_banjo.sif \
    java -Xmx5G -jar /usr/local/bin/banjo.jar \
    settingsFile=$PWD/banjo/settings.txt \
    threads=64 inputDirectory=$PWD/banjo/data \
    observationsFile=$datafile \
    outputDirectory=$PWD/banjo/output \
    variableCount=$ngenes 

echo
date
