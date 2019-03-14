
data = read.table("../data/toy-expXgenes.tsv")
net = cor(data)
write.table(net, "toy-cor.net")

