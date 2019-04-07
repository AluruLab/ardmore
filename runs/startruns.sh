#!/bin/bash

while read -r f || [[ -n "$f" ]]
do
    while read -r m || [[ -n "$m" ]]
    do
        /bin/bash $m/format.sh -f $f
        sbatch --export=datafile="$m"/data/format."$f" $m/run.sh
    done < methods.config
done < datafiles.config
