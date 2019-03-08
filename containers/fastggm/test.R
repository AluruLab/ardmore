library(FastGGM)
library(RcppParallel)

# requires samples*vars data

X <- read.table("data", header=FALSE, sep="\t"
outlist1 <- FastGGM(
