FROM tanyard/im:base 

ENV MPI_C_COMPILER /usr/local/bin/mpicc
ENV MPI_CXX_COMPILER /usr/local/bin/mpicxx
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV MPI_C /usr/local/bin/mpicc
ENV MPI_CXX /usr/local/bin/mpicxx


RUN sed -i \
  '/malloc_usable_size/a \\nstruct mallinfo {\n  int arena;\n  int ordblks; \n  int smblks;\n  int hblks;\n  int hblkhd;\n  int usmblks;\n  int fsmblks;  \n  int uordblks;\n int fordblks; \n int keepcost;};' \ 
  /usr/include/malloc.h


RUN mkdir -p /work/ && cd /work && \
    git clone https://github.com/srirampc/tinge.git && \
    cd tinge/ && \
    mkdir -p build.cm/ && cd build.cm/ && \
    cmake .. && \
    make && \
    cp tinge-mi /usr/local/bin && \
    cp residuals /usr/local/bin && \
    cd /work/tinge/tools/ && \
    chmod +x adj-compact adj-consensus adj-mi2cc adj-probes adj-prune adj-size adj2lab adj2sif csv2exp dream exp-select exp-size mcl2noa sif-compare sif2adj && \
    cp adj-compact adj-consensus adj-mi2cc adj-probes adj-prune adj-size adj2lab adj2sif csv2exp dream exp-select exp-size mcl2noa sif-compare sif2adj /usr/local/bin && \
    rm -rf /work/tinge/

