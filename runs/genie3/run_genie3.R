#!/usr/bin/env Rscript
library(GENIE3)
library(doParallel)
library(doRNG)
library(reshape2)

args = commandArgs(trailingOnly=TRUE)

X <- read.table(args[1], header=TRUE, sep="", row.names=1)
expts_data <- data.matrix(X, rownames.force=TRUE)
cat("Running genie3 on #cores : ", as.numeric(args[3]))
weightMatrix <- GENIE3(expts_data, nCores=as.numeric(args[3]))
linkList <- getLinkList(weightMatrix)
write.table(linkList, file=args[2])
