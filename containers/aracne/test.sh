#!/bin/bash

java -Xmx5G -jar /usr/local/bin/aracne.jar -e data/toy-genesXexp.exp -o . --tfs data/genes.txt --pvalue 1E-8 --seed 1 --calculateThreshold
java -Xmx5G -jar /usr/local/bin/aracne.jar -e data/toy-genesXexp.exp -o . --tfs data/genes.txt --pvalue 1E-8 --seed 1

