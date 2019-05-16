library(GeneNet)

#requires samplesXgenes
args = commandArgs(trailingOnly=TRUE)
run.ggm.network <- function(exp.fname, out.fname){
    X <- read.table(exp.fname, header=TRUE, sep=" ", row.names=1)
    expts_data <- data.matrix(X, rownames.force=TRUE)
   #exp_tab <- read.table(exp.fname , skip = 3, as.is = T)
   #rownames(exp_tab) <- exp_tab[,1]
   #dt <- dim(exp_tab)
   #exp_tab <- exp_tab[,3:dt[2]]
   #exp_tab <- t(exp_tab)
    inferred.pcor <- ggm.estimate.pcor(expts_data)
    write.table(inferred.pcor, out.fname)
    pvalues <- network.test.edges(inferred.pcor, plot=FALSE)
    write.table(pvalues[,c("node1", "node2", "pval")], paste(out.fname, "stats", sep="_"))
}

run.ggm.network(args[1], args[2])
