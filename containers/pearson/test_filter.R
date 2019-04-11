filter.pearson.network <- function(mat.fname, outf.name, n){
    mx = as.matrix(read.table(mat.fname))
    cvx = as.vector(mx)
    dvx = data.frame(s=rep(rownames(mx), each=dim(mx)[1]), 
                     t=rep(rownames(mx), dim(mx)[1]), 
                     wt=cvx, stringsAsFactors=FALSE)
    dvx = dvx[ dvx$s < dvx$t, ]
    dvxsrt = dvx[order(abs(dvx$wt), decreasing = TRUE)[1:n], ]
    write.table(dvxsrt, outf.name, sep="\t", row.names=FALSE)
}



args = commandArgs(trailingOnly=TRUE)
print(length(args))
if (length(args) == 3) {
    print(args)
    filter.pearson.network(args[1], args[2], as.integer(args[3]))
}
