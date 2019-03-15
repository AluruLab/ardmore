library(GeneNet)

run.ggm.network <- function(exp.fname, out.fname){
   exp_tab <- read.table(exp.fname , skip = 3, as.is = T)
   rownames(exp_tab) <- exp_tab[,1]
   dt <- dim(exp_tab)
   exp_tab <- exp_tab[,3:dt[2]]
   exp_tab <- t(exp_tab)
   inferred.pcor <- ggm.estimate.pcor(exp_tab)
   write.table(inferred.pcor, out.fname, sep = "\t")
}

run.ggm.network("../data/toy-genesXexp.tsv", "out-net.tsv")
