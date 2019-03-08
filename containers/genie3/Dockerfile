FROM tanyard/im:base

RUN mkdir -p /work/ && cd /work && \
    git clone https://github.com/vahuynh/GENIE3.git && \
    cd GENIE3/GENIE3_R_C_wrapper && \
    R CMD SHLIB GENIE3.c && \
    cp GENIE3.R /usr/local/bin

RUN R -e 'install.packages(c("reshape2","doRNG","doParallel"),  repos="https://cran.r-project.org")'
