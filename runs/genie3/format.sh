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

ngenes=$(head -n1 $filename | wc -w)
nsamples=$(tail -n +2 $filename | wc -l)
mkdir -p genie3/data
awk '
{ 
    for (i=1; i<=NF; i++)  {
        a[NR,i] = $i
    }
}
NF>p { p = NF }
END {    
    for(j=1; j<=p; j++) {
        str=a[1,j]
        for(i=2; i<=NR; i++){
            str=str"\t"a[i,j];
        }
        print str
    }
}' <(tail -n +2 $filename | cut -d ' ' -f 2-) >genie3/data/temp

paste <(head -n1 $filename | sed 's/"//g' | tr ' ' '\n') genie3/data/temp >genie3/data/temp2
formattedname=${filename##*/}
{ for ((i=1;i<$nsamples;i++)); do echo -n -e "S$i\t"; done; echo "S$nsamples"; cat genie3/data/temp2; } >genie3/data/format.$formattedname
rm genie3/data/{temp,temp2}
