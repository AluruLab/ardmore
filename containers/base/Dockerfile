FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt update

RUN apt install -y apt-utils \
    build-essential \
    curl \
    wget \
    time \
    cmake \
    gcc-8 \
    g++-8 \
    unzip vim \
    libopenblas-dev \
    libcurl4-openssl-dev \
    git \
    automake \
    autoconf \
    apt-transport-https \
    software-properties-common

RUN apt autoremove && apt clean && apt update 

RUN apt install -y  \
    libcurl4-gnutls-dev \
    libxml2-dev \
    libssl-dev 

RUN apt autoremove && apt clean && apt update 

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
    apt update && \
    apt install -y r-base r-base-dev

RUN apt update && \
    apt install -y python3.6 \
    python3-pip \
    default-jdk \
    ant 

RUN apt update && \
    apt install -y mpich libmpich-dev

RUN apt autoremove && apt clean && apt update 

RUN R -e 'install.packages("BiocManager",repos = "http://cran.us.r-project.org")'

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/jre
ENV PATH $PATH:/usr/lib/jvm/java-8-openjdk-amd64/jre/bin
ENV MPI_C_COMPILER /usr/bin/mpicc
ENV MPI_CXX_COMPILER /usr/bin/mpicxx
