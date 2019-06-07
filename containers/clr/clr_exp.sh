#!/bin/bash

expfname=$1
outfname=$2
tmpdata=`mktemp`
tmpoutput=`mktemp`

echo "TMP DATASET : " $tmpdata
echo "TMP OUTPUT  : " $tmpoutput
tail -n +4 $expfname | cut -f 3- | tr '\t' ',' > $tmpdata
clr --data $tmpdata --map $tmpoutput --bins 10 --spline 3
tail -n +4 $expfname | cut -f 1 | paste -sd'\t' > $outfname
tail -n +4 $expfname | cut -f 1 | paste -d "\t" - $tmpoutput  | tr ',' '\t' >> $outfname

rm -f $tmpdata $tmpoutput
