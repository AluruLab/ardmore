FROM tanyard/im:base

RUN mkdir -p /work/ && cd /work && \
    wget http://m3d.mssm.edu/network_inference/clr_v1.2.2.tar.gz && \
    tar -xf clr_v1.2.2.tar.gz && \
    cd CLRv1.2.2/Code && cp clr /usr/local/bin
