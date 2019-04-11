
args = commandArgs(trailingOnly=TRUE)
print(length(args))
if (length(args) == 2) {
    print(args)
    data = read.table(args[1])
    net = cor(data)
    write.table(net, args[2])
} else {
   data = read.table("../data/toy-expXgenes.tsv")
   net = cor(data)
   write.table(net, "toy-cor.net")
}

