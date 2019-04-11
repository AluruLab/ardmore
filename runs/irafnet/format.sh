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

mkdir -p irafnet/data
formattedname=${filename##*/}
cp $filename irafnet/data/format.$formattedname
