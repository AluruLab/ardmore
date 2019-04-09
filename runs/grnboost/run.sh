#!/bin/bash
#SBATCH --job-name=grnboost
#SBATCH --output=./grnboost/logs/log_%J.out
#SBATCH --error=./grnboost/logs/err_%J.err
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

mkdir -p grnboost/output

export JAVA_HOME=/usr/lib/jvm/java-1.8-openjdk/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
# ./build.sh
# singularity pull docker://tanyard/im:grnboost
# singularity run im_grnboost.sif mvn -version

currenttime=$(date "+%Y.%m.%d-%H.%M.%S")
singularity run $PWD/grnboost/im_grnboost.sif /usr/local/bin/spark/bin/spark-submit \
     --class org.aertslab.grnboost.GRNBoost \
     --master local[*] \
     --deploy-mode client \
     --jars /m2/repo/ml/dmlc/xgboost4j/0.83-SNAPSHOT/xgboost4j-0.83-SNAPSHOT.jar \
      /work/GRNBoost/target/scala-2.11/GRNBoost.jar \
     infer \
     -i $PWD/$datafile \
     -tf <(tail -n +2 $PWD/datafile | cut -d$'\t' -f 1) \
     -o $PWD/grnboost/output/${ngenes}.${nexpts}-${currenttime} \
     -p eta=0.01 \
     -p max_depth=3 \
     -p colsample_bytree=0.1 \
     --truncate 100000

echo
date
