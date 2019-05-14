#!/bin/bash

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
# export PATH=$LD_LIBRARY_PATH:/usr/local/lib/
# ./build.sh
# singularity pull docker://tanyard/im:grnboost
# singularity run im_grnboost.sif mvn -version
#     --total-executor-cores 7 \
/usr/bin/time -f "%e %t %P %K"  singularity run im_grnboost.sif /usr/local/bin/spark/bin/spark-submit \
     --class org.aertslab.grnboost.GRNBoost \
     --driver-memory 32g \
     --executor-memory 8g \
     --master local[16] \
     --deploy-mode client \
     --jars /m2/repo/ml/dmlc/xgboost4j/0.83-SNAPSHOT/xgboost4j-0.83-SNAPSHOT.jar \
      /work/GRNBoost/target/scala-2.11/GRNBoost.jar \
     infer \
     -i ./exp-data.tsv \
     -tf ./genes \
     -o ./grnboost-net-small \
     -p eta=0.01 \
     -p max_depth=3 \
     -p colsample_bytree=0.1 \
     --truncate 100000
