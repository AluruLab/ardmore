#!/bin/bash

module load singularity-3.0
while read -r f || [[ -n "$f" ]]
do
 if [[ -f $f ]]; then
    while read -r m || [[ -n "$m" ]]
    do
      if [[ -d $m ]]; then
        echo "Method: $m"
        echo "Dataset: $f"
        formattedname=${f##*/}
        /bin/bash $m/format.sh -f $f
        mkdir -p $m/logs
		if [ ! -f "$m/im_$m.sif" ]
		then
			cd $m
			singularity pull docker://tanyard/im:$m
			cd ../
		fi	
        sbatch --export=datafile="$m"/data/format."$formattedname" $m/run.sh
      fi
    done < methods.config
 fi
done < datafiles.config
