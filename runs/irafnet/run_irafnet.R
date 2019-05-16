library(iRafNet)

args = commandArgs(trailingOnly=TRUE)
X <- read.table(args[1], header=TRUE, sep=" ", row.names=1)
Y <- as.matrix(X, rownames.force=TRUE)
#W <- abs(matrix(rnorm(p*p), p, p))
#Give unit weight prior to all edges
W <- matrix(1L, nrow=dim(Y)[2], ncol=dim(Y)[2])

out <- iRafNet(Y, W, genes.name=colnames(Y))
write.table(out, args[2])
