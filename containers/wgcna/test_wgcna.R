library(WGCNA)

run.wgcna <- function(exp.fname, out.fname, softPower=1){
   exp_tab <- read.table(exp.fname , skip = 3, as.is = T)
   rownames(exp_tab) <- exp_tab[,1]
   dt <- dim(exp_tab)
   exp_tab <- exp_tab[,3:dt[2]]
   exp_tab <- t(exp_tab)
   # computed.cor <- cor(exp_tab)
   # write.table(computed.cor, out.fname, sep = "\t")
   adj = adjacency(df,type = "unsigned", power = softPower)
   # TOM = TOMsimilarityFromExpr(df, power=softPower, TOMType="unsigned")
   # write.table(TOM, file=outfile)
}

args = commandArgs(trailingOnly=TRUE)
options(stringsAsFactors=FALSE)
print(length(args))
print(args)
if (length(args) == 2) {
    run.wgcna(args[1], args[2])
}

if (length(args) >= 3) {
    threads <- as.numeric(args[3])
    enableWGCNAThreads(threads)
}
if (length(args) == 3) {
    run.wgcna(args[1], args[2])
}

