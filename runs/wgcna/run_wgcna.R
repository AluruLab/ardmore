library(WGCNA)

args = commandArgs(trailingOnly=TRUE)
datafile <- args[1]
outfile <- args[2]
threads <- args[3]
options(stringsAsFactors = FALSE);
enableWGCNAThreads(nthreads=threads)
# requires samples X genes
df <- read.csv(datafile, sep = "")
# TODO : pick soft threshold
softPower = 8
adj= adjacency(df,type = "unsigned", power = softPower)
TOM <- TOMsimilarityFromExpr(df, power=softPower, TOMType="unsigned")
write.matrix(TOM, file=outfile)
