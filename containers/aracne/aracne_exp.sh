#!/bin/bash

if [ "$#" -lt 5 ]; then
     echo "Usage: arace_exp <INPUT_EXP> <BOOTSTRAP_DIR> <OUTPUT_FILE> <NO_THREADS> <NO_BOOTSTRAP> [<MI_THLD_FLAG>]"
     exit 0
fi

aracne_jar=/nv/hswarm1/schockalingam6/data/tanyard/simulations/containers/aracne/aracne.jar
expfile=$1
bootdir=$2
outfile=$3
nthreads=$4
nbsp=$5
if [ "$#" -gt 5 ]; then
   mith="N"
else
   mith="Y"
fi

printf "input exp file : %s\n" $expfile 
printf "bootstrap dir  : %s\n" $bootdir
printf "output file    : %s\n" $outfile
printf "no. threads    : %s\n" $nthreads
printf "no. bootstrap  : %s\n" $nbsp
printf "mi. threshold  : %s\n" $mith

texpfile=`mktemp`
tfsfile=`mktemp`
{ head -n 1 $expfile ; tail -n +4 $expfile ; } | cut -f 1,3- > $texpfile
tail -n +4 $expfile | cut -f 1 > $tfsfile

# rseed=`echo $RANDOM`
rseed=`date '+%s'`

printf "tfs file       : %s\n" $tfsfile
printf "texp file      : %s\n" $texpfile
printf "rseed          : %s\n" $rseed
if [[ "$mith" == "Y" ]]; then
  if [[ -d $bootdir ]]; then
     rm -rf $bootdir/bootstrapNetwork_*.txt  $bootdir/miThreshold_p5E-9_*.txt $bootdir/network.txt
  else
     mkdir -p $bootdir
  fi

  echo java -Xmx125G -jar ${aracne_jar} -e $texpfile -o $bootdir --tfs $tfsfile --pvalue 5E-9 --threads $nthreads --seed $rseed --calculateThreshold
  java -Xmx125G -jar ${aracne_jar} -e $texpfile -o $bootdir --tfs $tfsfile --pvalue 5E-9 --threads $nthreads --seed $rseed --calculateThreshold
else
  echo "Skipping generation of MI threshold"
fi

for ((i = 1 ; i <= $nbsp ; i++)); do
    echo bootstrap round $i
    echo java -Xmx125G -jar ${aracne_jar} -e $texpfile -o $bootdir --tfs $tfsfile --pvalue 5E-9 --threads $nthreads
    java -Xmx125G -jar ${aracne_jar} -e $texpfile -o $bootdir --tfs $tfsfile --pvalue 5E-9 --threads $nthreads
done

echo java -Xmx125G -jar ${aracne_jar} -o $bootdir --consolidate
java -Xmx125G -jar ${aracne_jar} -o $bootdir --consolidate

mv $bootdir/network.txt $outfile

rm $texpfile $tfsfile

