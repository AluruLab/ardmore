library(Rcpp)
library(RcppParallel)
library(FastGGM)
library(stringr)

run.fastggm <- function(exp.fname, out.fname, softPower=1){
   exp_tab <- read.table(exp.fname , skip = 3, as.is = T)
   rownames(exp_tab) <- exp_tab[,1]
   dt <- dim(exp_tab)
   exp_tab <- exp_tab[,3:dt[2]]
   exp_tab <- t(exp_tab)
   out_net <- FastGGM_Parallel(exp_tab)
   out.pval.fname <- str_replace(out.fname, ".mat", "-pval.mat")
   write.table(out_net$precision, file=out.fname)
   write.table(out_net$p_precision, file=out.pval.fname)
}

args = commandArgs(trailingOnly=TRUE)
options(stringsAsFactors=FALSE)
print(length(args))
print(args)
if (length(args) >= 3) {
    threads = as.numeric(args[3])
    setThreadOptions(numThreads=threads)
} else {
    #enableWGCNAThreads()
}
if (length(args) == 2) {
    run.fastggm(args[1], args[2])
}
if (length(args) == 3) {
    run.fastggm(args[1], args[2])
}
#
