FROM tanyard/im:base

RUN R -e 'install.packages("Rcpp",repos = "http://cran.us.r-project.org")' 

RUN mkdir -p /work/ && \
    cd /work && \
    wget https://github.com/srirampc/fs/archive/master.tar.gz -O fs-master.tar.gz && \
    R -e 'install.packages("fs-master.tar.gz", repos = NULL, type="source")'

RUN R -e 'install.packages("devtools",repos = "http://cran.us.r-project.org")'

RUN R -e 'library(devtools); install_github("shruti1995/tigress")'
