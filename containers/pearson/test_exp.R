run.pearson.network <- function(exp.fname, out.fname){
   exp_tab <- read.table(exp.fname , skip = 3, as.is = T)
   rownames(exp_tab) <- exp_tab[,1]
   dt <- dim(exp_tab)
   exp_tab <- exp_tab[,3:dt[2]]
   exp_tab <- t(exp_tab)
   computed.cor <- cor(exp_tab)
   write.table(computed.cor, out.fname, sep = "\t")
}

args = commandArgs(trailingOnly=TRUE)
print(length(args))
if (length(args) == 2) {
    print(args)
    run.pearson.network(args[1], args[2])
}
