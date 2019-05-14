#!/bin/bash
#SBATCH --job-name=grnboost
#SBATCH --output=./grnboost/logs/oe_%J.log
#SBATCH --error=./grnboost/logs/oe_%J.log
#SBATCH --ntasks=64
#SBATCH --time=24:00:00

ngenes=$(tail -n +1 $datafile | wc -l)
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
tflist=${PWD}/${datafile}.genes
tail -n +1 $PWD/$datafile | cut -d$'\t' -f 1 > $tflist
echo "Datafile : $PWD/$datafile"
echo "TFList : $tflist"

/usr/bin/time -f "$ngenes $nexpts %e %S %P" singularity run $PWD/grnboost/im_grnboost.sif /usr/local/bin/spark/bin/spark-submit \
     --class org.aertslab.grnboost.GRNBoost \
     --driver-memory 32g \
     --executor-memory 8g \
     --master local[64] \
     --deploy-mode client \
     --jars /m2/repo/ml/dmlc/xgboost4j/0.83-SNAPSHOT/xgboost4j-0.83-SNAPSHOT.jar \
      /work/GRNBoost/target/scala-2.11/GRNBoost.jar \
     infer \
     -i $PWD/$datafile \
     -tf $tflist \
     -o $PWD/grnboost/output/${ngenes}X${nexpts}-${currenttime} \
     -p eta=0.01 \
     -p max_depth=3 \
     -p colsample_bytree=0.1 \
     --truncate 100000
