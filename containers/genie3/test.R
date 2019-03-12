library(GENIE3)
library(grndata)


#expr.matrix <- read.expr.matrix("data.txt", form="rows.are.samples")
#weight.matrix <- GENIE3(expr.matrix, nCores=16)

yeast.matrix <- getData(datasource.name="gnw2000",getNet=FALSE)
# TODO: CONVERT FROM GRNDATA TO GENIE3 FORMAT : yeast.matrix
weight.matrix <- GENIE3(yeast.matrix, nCores=16)
link.list <- get.link.list(weight.matrix)
write.table(link.list, file="weights.csv", sep=',', col.names=NA)

