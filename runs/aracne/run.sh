#!/bin/bash
#SBATCH --job-name=aracne
#SBATCH --output=./aracne/logs/log_%J.out
#SBATCH --error=./aracne/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd

ngenes=$(tail -n +2 $datafile | wc -l)
nexpts=$(head -n 1 $datafile | wc -w)
((nexpts--))
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts"
echo "Running aracne on #cores : 64"

module load singularity-3.0

mkdir -p aracne/output
outputdir=aracne/output/$(date "+%Y.%m.%d-%H.%M.%S")
mkdir $outputdir

singularity exec \
	$PWD/aracne/im_aracne.sif \
	time java -Xmx5G -jar /usr/local/bin/aracne.jar \
	-e $datafile -o $outputdir \
	--tfs <(tail -n +2 $PWD/$datafile | cut -d$'\t' -f 1) \
	--pvalue 1E-8 --seed 1 --calculateThreshold

singularity exec \
	$PWD/aracne/im_aracne.sif \
	time java -Xmx5G -jar /usr/local/bin/aracne.jar \
	-e $datafile -o $outputdir \
	--tfs <(tail -n +2 $PWD/$datafile | cut -d$'\t' -f 1) \
	--pvalue 1E-8 --seed 1 --threads 64

echo
date
