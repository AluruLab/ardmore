library("tigress")
library("parallel")

#data(ecoli)
#
#dim(ecoli$exp)
#ecoli$exp[1:5,1:3]
#genenames <- rownames(ecoli$exp)
#nreg <- nrow(ecoli$reg)
#tfindices <- sort(unique(ecoli$reg[,1]))
#tfnames <- genenames[tfindices]
#ntf <- length(tfindices)
#targetindices <- sort(unique(ecoli$reg[,2]))
#targetnames <- genenames[targetindices]
#ntarget <- length(targetindices)
#nstepsLARS = 20
#edgepred <- tigress(t(ecoli$exp), tflist=tfnames, targetlist=targetnames, nstepsLARS = nstepsLARS)

args = commandArgs(trailingOnly=TRUE)
X <- read.table(args[1], header=TRUE, sep=" ", row.names=1)
Y <- data.matrix(X, rownames.force=TRUE)
output <- tigress(Y, verb=TRUE, usemulticore=as.numeric(args[3]))
write.table(output, args[2])
