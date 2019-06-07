library(tigress)

run.tigress <- function(exp.fname, out.fname, nsp=8000, vb=FALSE){
   exp_tab <- read.table(exp.fname , skip = 3, as.is = T)
   rownames(exp_tab) <- exp_tab[,1]
   dt <- dim(exp_tab)
   exp_tab <- exp_tab[,3:dt[2]]
   exp_tab <- t(exp_tab)
   computed.mat <- tigress(exp_tab, alpha=0.4, nstepsLARS=3, nsplit=nsp, 
                           normalizeexp=FALSE, allsteps=FALSE, usemulticore=28L, 
                           recursive=FALSE, verb=vb)
   write.table(computed.mat, out.fname, sep = "\t")
}

args = commandArgs(trailingOnly=TRUE)
print(length(args))
if (length(args) == 2) {
    print(args)
    run.tigress(args[1], args[2])
} else {
  if (length(args) == 3) {
    print(args)
    run.tigress(args[1], args[2], as.numeric(args[3]))
  } else {
   if (length(args) == 4) {
    print(args)
    run.tigress(args[1], args[2], as.numeric(args[3]), TRUE)
   }
  }
}
