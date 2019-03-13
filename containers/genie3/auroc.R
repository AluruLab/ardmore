library(pROC)
library(netbenchmark)

yeast.matrix <- getData(datasource.name="gnw2000",getNet=TRUE)
true_net <- yeast.matrix[[2]]
genes <- rownames(true_net)

df <- read.table("weights_genie3.csv", header=TRUE)
g1 <- match(df$regulatoryGene, genes)
g2 <- match(df$targetGene, genes)
df2 <- data.frame("g1"=g1, "g2"=g2, "weight"=df$weight)

pred_net <- matrix(0L, nrow = dim(true_net)[1], ncol = dim(true_net)[2]) 
pred_net[cbind(df2[["g1"]], df2[["g2"]])] <- df2[["weight"]]

tbl <- evaluate(pred_net, true_net)
write_matrix(tbl, file="confusion_matrix_netbenchmark")
auc_ <- auroc(tbl)
aupr_ <- aupr(tbl)
sprintf("AUC score with netbenchmark = %e, AUPR score = %e\n", auc_, aupr_)
auc_ <- auc(roc(response=c(true_net), predictor=c(pred_net)))
sprintf("AUC score with proc = %e\n", auc_)



