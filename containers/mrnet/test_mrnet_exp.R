library(minet)

run.minet <- function(exp.fname, out.fname) {
   exp_tab <- read.table(exp.fname , skip = 3, as.is = T)
   rownames(exp_tab) <- exp_tab[,1]
   dt <- dim(exp_tab)
   exp_tab <- exp_tab[,3:dt[2]]
   exp_tab <- t(exp_tab)
   mim = minet::build.mim(exp_tab,estimator="spearman")
   net = minet::mrnet(mim)
   write.table(net, file=out.fname)
}

args = commandArgs(trailingOnly=TRUE)
options(stringsAsFactors=FALSE)
print(length(args))
print(args)
if (length(args) >= 2) {
    run.minet(args[1], args[2])
}

