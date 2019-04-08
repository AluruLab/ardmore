#!/bin/bash

while read -r f || [[ -n "$f" ]]
do
    while read -r m || [[ -n "$m" ]]
    do
	echo "Method: $m"
	echo "Dataset: $f"
	formattedname=${f##*/}
        /bin/bash $m/format.sh -f $f
        sbatch --export=datafile="$m"/data/format."$formattedname" $m/run.sh
    done < methods.config
done < datafiles.config
