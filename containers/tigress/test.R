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

X <- read.table("data/toy-expXgenes.tsv", header=TRUE, sep=" ", row.names=1)
Y <- data.matrix(X, rownames.force=TRUE)
output <- tigress(Y, verb=TRUE, usemulticore=TRUE)
write.table(output, "tigress.out")


