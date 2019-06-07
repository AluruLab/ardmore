library(WGCNA)

run.wgcna <- function(exp.fname, out.fname, softPower=1){
   exp_tab <- read.table(exp.fname , skip = 3, as.is = T)
   rownames(exp_tab) <- exp_tab[,1]
   dt <- dim(exp_tab)
   exp_tab <- exp_tab[,3:dt[2]]
   exp_tab <- t(exp_tab)
   # computed.cor <- cor(exp_tab)
   # write.table(computed.cor, out.fname, sep = "\t")
   sth = pickSoftThreshold(exp_tab, TRUE, moreNetworkConcepts=TRUE)
   ftx = sth$fitIndices
   ftx$Fitsq = -sign(ftx[,3])*ftx[,2]
   print(ftx)
   adjm = adjacency(exp_tab,type = "unsigned", power = softPower)
   # print(dim(adjm))
   #TOM = TOMsimilarity(adjm, TOMType="unsigned", suppressTOMForZeroAdjacencies=TRUE)
   #rownames(TOM) = rownames(adjm)
   #colnames(TOM) = colnames(adjm)
   write.table(adjm, file=out.fname)
}

args = commandArgs(trailingOnly=TRUE)
options(stringsAsFactors=FALSE)
print(length(args))
print(args)
if (length(args) >= 3) {
    threads = as.numeric(args[3])
    enableWGCNAThreads(threads)
} else {
    #enableWGCNAThreads()
}
if (length(args) == 2) {
    run.wgcna(args[1], args[2])
}
if (length(args) == 3) {
    run.wgcna(args[1], args[2])
}
if (length(args) == 4) {
    run.wgcna(args[1], args[2], as.numeric(args[4]))
}
#
