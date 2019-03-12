FROM tanyard/im:base

RUN R -e 'install.packages(c("reshape2","doRNG","doParallel"),  repos="https://cran.r-project.org")'


RUN R -e 'BiocManager::install("GENIE3", version = "3.8")'