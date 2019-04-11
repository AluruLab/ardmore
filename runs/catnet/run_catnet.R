library(catnet)

args = commandArgs(trailingOnly=TRUE)
X <- read.table(args[1], header=TRUE, row.names=1)
eval <- cnSearchSA(data=X, maxParentSet=2, numThreads=args[3], echo=TRUE)
bnet = cnFindBIC(object=eval, numsamples=nrow(X))
fileConn <- file(args[2])
writeLines(cnPlotProb(bnet))
close(fileConn)
