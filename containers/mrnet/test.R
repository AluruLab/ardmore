library(minet)


data = read.table("../data/toy-expXgenes.tsv")
mim = minet::build.mim(data,estimator="spearman")
net = minet::mrnet(mim)
write.table(net, "net.tsv")
