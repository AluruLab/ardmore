library(WGCNA)

args = commandArgs(trailingOnly=TRUE)
datafile <- args[1]
outfile <- args[2]
threads <- as.numeric(args[3])
options(stringsAsFactors = FALSE)
enableWGCNAThreads(threads)
# requires samples X genes
df <- read.csv(datafile, sep = "")
# TODO : pick soft threshold
softPower = 8
adj= adjacency(df,type = "unsigned", power = softPower)
TOM <- TOMsimilarityFromExpr(df, power=softPower, TOMType="unsigned")
write.table(TOM, file=outfile)
