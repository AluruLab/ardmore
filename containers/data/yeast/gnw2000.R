data <- getData(datasource.name = "gnw2000", getNet = TRUE)

expts <- data[[1]]
dime <- dim(expts)
net <- data[[2]]
sds <- apply(expts, 2, sd)
local.noise <- 20
global.noise <- 10
noise.l <- runif(1,local.noise*0.8,local.noise*1.2)

exptsn <- apply(expts, 2, function(x, noise=noise.l){
  if(sd(x)!=0)
    s.d <- runif(1,noise*0.8,noise*1.2)*sd(x)/100
  else
    s.d <- runif(1,0.01,0.15)
  n <- rnorm(length(x),mean=0,sd=s.d)
  return (x+n)
})

noise.g <- runif(1,global.noise*0.8,global.noise*1.2)
Gnoise <- matrix(rlnorm(dim(exptsn)[1]*dime[2],meanlog=0,
                        sdlog=mean(sds)*noise.g/100), 
                 dim(exptsn)[1], dime[2])
exptsnn <- exptsn + Gnoise

write.table(exptsnn, file="~/Desktop/gnw2000_noise")