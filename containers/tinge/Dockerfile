FROM tanyard/im:base

RUN mkdir -p /work/ && cd /work && \
    wget https://swarm.workshop.perforce.com/projects/perforce_software-jam/download/main/jam-2.6.tar && \
    tar -xvf jam-2.6.tar && cd jam-2.6/ && \
    make all && mv jam0 jam && \
    cp jam /usr/local/bin

RUN mkdir -p /work/ && cd /work && \
    wget https://alurulab.cc.gatech.edu/sites/all/projectFiles/tinge_gena/tinge-1.062.tar.bz2 && \
    tar xfj tinge-1.062.tar.bz2 && \
    cd TINGe-1.062/ && ls -la build/* && \
    sh x-build install && \
    cp bin/tinge-mi /usr/local/bin
