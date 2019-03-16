#!/bin/bash

/usr/local/bin/clr --data <(tail -n +2 data/toy-genesXexp.tsv | cut -d ' ' -f 2-) --map clrnet.output --bins 10 --spline 3
