#!/usr/bin/env Rscript
library(GENIE3)

args = commandArgs(trailingOnly=TRUE)

X <- read.table(args[1], header=TRUE, sep=" ", row.names=1)
expts_data <- data.matrix(X, rownames.force=TRUE)
weightMatrix <- GENIE3(expts_data, nCores=args[3])
linkList <- getLinkList(weightMatrix)
write.table(linkList, file=args[2])
