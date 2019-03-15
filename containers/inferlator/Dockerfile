FROM tanyard/im:base

ENV INF_PATH /usr/local/bin

RUN R -e 'install.packages(c("ggplot2", "reshape2", "Matrix", "gplots", "markdown", "inline", "knitr", "elasticnet", "corpcor", "nnls"), repos="https://cran.r-project.org")'

RUN mkdir -p /work/ && cd work && \
    git clone https://github.com/srirampc/Inferelator.git && \
    cd Inferelator && \
    cp *.R /usr/local/bin && \
    cp -R R_scripts /usr/local/bin && \
    cp -R input /usr/local/bin && \
    cp -R jobs /usr/local/bin

