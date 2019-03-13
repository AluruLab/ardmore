library(GENIE3)
library(grndata)


#expr.matrix <- read.expr.matrix("data.txt", form="rows.are.samples")
#weight.matrix <- GENIE3(expr.matrix, nCores=16)

yeast.matrix <- getData(datasource.name="gnw2000",getNet=TRUE)
expts <- yeast.matrix[[1]]
true_net <- yeast.matrix[[2]]

expts_data <- matrix(unlist(expts), nrow=length(expts), byrow=TRUE)
rownames(expts_data) <- names(expts)
colnames(expts_data) <- paste("Sample", 1:2000)
# TODO: CONVERT FROM GRNDATA TO GENIE3 FORMAT : yeast.matrix
weightMatrix <- GENIE3(expts_data, nCores=16)
linkList <- getLinkList(weightMatrix)
write.table(link.list, file="weights.csv", sep=',', col.names=NA)
