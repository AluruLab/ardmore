library(GENIE3)
library(doParallel)
library(doRNG)
library(reshape2)

run.genie3 <- function(exp.fname, out.fname, nc=1){
   exp_tab <- read.table(exp.fname , skip = 3, as.is = T)
   rownames(exp_tab) <- exp_tab[,1]
   dt <- dim(exp_tab)
   exp_tab <- exp_tab[,3:dt[2]]
   # computed.cor <- cor(exp_tab)
   # write.table(computed.cor, out.fname, sep = "\t")
   XD <- data.matrix(exp_tab, rownames.force=TRUE)
   cat("Running genie3 on #cores : ", nc, "\n")
   weightMatrix <- GENIE3(XD, nCores=nc)
   linkList <- getLinkList(weightMatrix)
   write.table(linkList, file=out.fname, sep='\t', row.names=FALSE)
}

args = commandArgs(trailingOnly=TRUE)
options(stringsAsFactors=FALSE)
print(length(args))
print(args)
threads=if (length(args) >= 3) {
    as.numeric(args[3])
} else {
    1
}
if (length(args) == 2) {
    run.genie3(args[1], args[2], threads)
}
if (length(args) == 3) {
    run.genie3(args[1], args[2], threads)
}
#
