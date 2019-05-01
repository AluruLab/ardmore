FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y apt-utils \
    build-essential \
    curl wget \
    cmake gcc \
    unzip vim \
    libopenblas-dev \
    libcurl4-openssl-dev \
    git \
    automake autoconf \
    software-properties-common
RUN apt-get clean && apt-get purge

RUN apt-get install -y apt-transport-https
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu xenial-cran35/' && \
    apt-get update && \
    apt-get install -y r-base r-base-dev

RUN mkdir -p /work/ && cd /work && \
    wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz && \
    gzip -dc openmpi-4.0.0.tar.gz | tar -x  && \
    cd openmpi-4.0.0  && \
    ./configure  && \
    make  && \
    make install && \
    cd ..  && \
    rm -rf openmpi-4.0.0

RUN R -e 'install.packages("BiocManager",repos = "http://cran.us.r-project.org")'

ENV MPI_C_COMPILER /usr/local/bin/mpicc
ENV MPI_CXX_COMPILER /usr/local/bin/mpicxx

RUN apt-get install -y time
RUN apt-get install -y libcurl4-gnutls-dev \
    libxml2-dev \
    libssl-dev

RUN apt-get install -y default-jdk \
    ant 

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6 \
    python3-pip

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk/jre
ENV PATH $PATH:/usr/lib/jvm/java-8-openjdk/jre/bin
ENV MPI_C_COMPILER /usr/local/bin/mpicc
ENV MPI_CXX_COMPILER /usr/local/bin/mpicxx
