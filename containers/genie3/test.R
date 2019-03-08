source("/usr/local/bin/GENIE3.R")

expr.matrix <- read.expr.matrix("data.txt", form="rows.are.samples")
weight.matrix <- GENIE3(expr.matrix, ncores=16)
link.list <- get.link.list(weight.matrix)
write.table(link.list, file="weights.csv", sep=',', col.names=NA)

