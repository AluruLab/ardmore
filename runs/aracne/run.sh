#!/bin/bash
#SBATCH --job-name=aracne
#SBATCH --output=./aracne/logs/log_%J.out
#SBATCH --error=./aracne/logs/err_%J.err
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

date
pwd
#sbatch --export=infile='',genes='' slurm.sh

ngenes=$(tail -n +2 $datafile | wc -l)
nexpts=$(head -n 1 $datafile | wc -w)
((nexpts--))
echo "Number of genes : $ngenes"
echo "Number of expts : $nexpts"

module load singularity-3.0

mkdir -p aracne/output

#-B $PWD/aracne/data:/work/input,$PWD/aracne/output \

singularity -vvvv exec \
	#-B $PWD/aracne/data:/work/input,$PWD/aracne/output:/work/output \
	aracne/im_aracne.sif \
	java -Xmx5G -jar /usr/local/bin/aracne.jar \
	-e $datafile -o $PWD/aracne/output \
	--tfs <(tail -n +2 $PWD/$datafile | cut -d ' ' -f 1) \
	--pvalue 1E-8 --seed 1 --calculateThreshold

singularity exec aracne/im_aracne.sif java -Xmx5G -jar /usr/local/bin/aracne.jar -e $datafile -o aracne/output --tfs <(tail -n +2 $datafile | cut -d ' ' -f 1) --pvalue 1E-8 --seed 1 --threads 64

echo
date
