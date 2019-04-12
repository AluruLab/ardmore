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
powers = c(c(1:10), seq(from = 12, to=20, by=2))
sft = pickSoftThreshold(df, dataIsExpr = TRUE, powerVector = powers, corFnc = cor, corOptions = list(use = 'p'), networkType = "unsigned")
softPower = sft$powerEstimate

adj = adjacency(df, type = "unsigned", power = softPower)

#Pick hard threshold from weighted similarity matrix
ht = pickHardThreshold.fromSimilarity(adj)
hardcut = ht$cutEstimate

write.table(adj, file=outfile)
cat("Hard cut estimate : ", hardcut)
write(hardcut, file=outfile, append=TRUE)
