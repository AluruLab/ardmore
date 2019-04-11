library(catnet)

X <- read.table("../data/toy/toy-expXgenes.tsv", header=TRUE, row.names=1)
eval <- cnSearchSA(X, maxParentSet=2)
net <- cnFindBIC(object=eval, numsamples=nrow(X))
fileConn <- file(args[2])
writeLines(cnPlotProb(bnet))
close(fileConn)


