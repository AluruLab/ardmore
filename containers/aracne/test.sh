#!/bin/bash

java -Xmx5G -jar aracne.jar -e data.txt -o . --tfs genes.txt --pvalue 1E-8 --seed 1 --calculateThreshold
java -Xmx5G -jar aracne.jar -e data.txt -o . --tfs genes.txt --pvalue 1E-8 --seed 1

