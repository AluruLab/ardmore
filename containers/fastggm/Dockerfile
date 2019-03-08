FROM tanyard/im:base

RUN mkdir -p /work/ && cd /work && \
    wget http://www.pitt.edu/~wec47/Files/FastGGM.tar.gz

RUN R -e 'install.packages(c("Rcpp", "RcppParallel"), repos="https://cran.us.r-project.org")'

RUN cd /work && \
    R -e 'install.packages(pkgs="FastGGM.tar.gz", repos=NULL, type="source")'
