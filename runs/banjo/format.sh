#!/bin/bash

unset filename
while getopts ":f:" opt
do
    case $opt in 
        f ) 
            filename=$OPTARG
            ;;
        \? ) 
            echo "Usage format.sh [-f filename]"
            exit
            ;;
        : )
            echo "Invalid option: -$OPTARG requires an argument"
            exit
            ;;
    esac
done
shift $((OPTIND -1))

if [[ -z "$filename" ]]
then
    echo "Usage format.sh [-f filename]"
    exit
fi

mkdir -p banjo/data
formattedname=${filename##*/}
cp $filename banjo/data/format.$formattedname
banjoin=banjo/data/format.$formattedname
tail -n +2 $banjoin | cut -d ' ' -f 2- >banjo/data/t
rm $banjoin && mv banjo/data/t $banjoin
