#!/usr/bin/env Rscript
library(minet)

# requires samplesXgenes
args = commandArgs(trailingOnly=TRUE)
data = read.table(args[1])
mim = minet::build.mim(data,estimator="spearman")
net = minet::mrnet(mim)
write.table(net, args[2])
