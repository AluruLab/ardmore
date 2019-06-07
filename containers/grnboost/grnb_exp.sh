#!/bin/bash

if [ "$#" -lt 3 ]; then
     echo "Usage: grn_exp <INPUT_EXP> <OUTPUT_FILE> <NEDGES>"
     exit 0
fi

module load gcc
module load java/1.8.0_66
export JAVA_HOME=/usr/local/pacerepov2/java/1.8.0_172/

expfile=$1
outfile=$2
nedges=$3

echo   "LD_LIBRARY_PATH: " $LD_LIBRARY_PATH
printf "input exp file : %s\n" $expfile 
printf "output file    : %s\n" $outfile

texpfile=`mktemp`
tfsfile=`mktemp`
tail -n +4 $expfile | cut -f 1,3- > $texpfile
tail -n +4 $expfile | cut -f 1 > $tfsfile

printf "tfs file       : %s\n" $tfsfile
printf "texp file      : %s\n" $texpfile

/usr/bin/time -f "%e %t %P %K"  ~/data/spark/bin/spark-submit \
     --class org.aertslab.grnboost.GRNBoost \
     --driver-memory 32g \
     --executor-memory 12g \
     --master local[16] \
     --deploy-mode client \
     --jars ~/.m2/repository/ml/dmlc/xgboost4j/0.90/xgboost4j-0.90.jar \
      ~/data/GRNBoost/target/scala-2.11/GRNBoost.jar \
     infer \
     -i $texpfile \
     -tf $tfsfile \
     -o $outfile \
     -p eta=0.01 \
     -p max_depth=3 \
     -p colsample_bytree=0.1 \
     --truncate $nedges

rm $texpfile $tfsfile
if [[ -f ${outfile}.result.tsv ]]; then
	mv  ${outfile}.result.tsv $outfile
else
	rm -rf ${outfile}.result/
fi
