#!/bin/bash

module load singularity-3.0

for f in `cat datafiles.config`
do
    if [[ -f $f ]]; then
        for m in `cat methods.config`
        do
            if [[ -d $m ]]; then
              echo "Method: $m"
              echo "Dataset: $f"
              formattedname=${f##*/}
              /bin/bash $m/format.sh -f $f
              mkdir -p $m/logs
              if [[  -f "$m/im_$m.sif" ]]; then
                 echo "Skipping singularity pull.."
              else
                 cd $m
                 singularity pull docker://tanyard/im:$m
                 cd ../
              fi
              sbatch --export=datafile="$m"/data/format."$formattedname" $m/run.sh
           fi
        done
    fi
done
