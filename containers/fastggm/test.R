library(FastGGM)
library(RcppParallel)

# requires samples*vars data
setThreadOptions(numThreads=4)
X <- read.table("data/toy-expXgenes.tsv", header=TRUE, sep=" ", row.names=1)
Y <- data.matrix(X, rownames.force=TRUE)
outlist1 <- FastGGM_Parallel(Y)
