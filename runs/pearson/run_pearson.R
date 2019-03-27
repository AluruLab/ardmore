#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)
data = read.table(args[1])
net = cor(data)
write.table(net, "pearson.out")
