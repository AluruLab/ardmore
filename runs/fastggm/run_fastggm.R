#!/usr/bin/env Rscript

library(FastGGM)
library(RcppParallel)

args = commandArgs(trailingOnly=TRUE)
# requires samples*vars data
cat("Running fastggm on #cores : ", as.numeric(args[3]))
setThreadOptions(numThreads=as.numeric(args[3]))
X <- read.table(args[1], header=TRUE, sep=" ", row.names=1)
Y <- data.matrix(X, rownames.force=TRUE)
outlist1 <- FastGGM_Parallel(Y)
write.table(outlist1, args[2])
