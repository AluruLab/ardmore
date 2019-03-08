FROM tanyard/im:base

RUN mkdir -p /work/ && cd /work && \
    git clone https://github.com/califano-lab/ARACNe-AP.git && \
    cd ARACNe-AP && ant main && \
    cp dist/aracne.jar /usr/local/bin
