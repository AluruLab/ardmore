FROM alpine:3.9

RUN echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories

# python, R, gcc
RUN apk add --update \
    linux-headers \
    python3 \
    python3-dev \
    R \
    R-dev \
    R-doc \
    build-base \
    automake \
    autoconf \
    m4 \
    git \
    libgit2 \
    libgit2-dev \
    libuv \
    libuv-dev \
    openjdk8 \
    openjdk8-jre \
    apache-ant \
    cmake \
    && rm -rf /var/cache/apk/*

# make sure pip is available
RUN python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
    if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 /usr/bin/python; fi && \
    rm -r /root/.cache

# install open mpi 
RUN mkdir -p /work/ && cd /work/ && wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz && \
    gzip -dc openmpi-4.0.0.tar.gz | tar -x  && \
    cd openmpi-4.0.0  && \
    ./configure  && \
    make  && \
    make install && \
    cd ..  && \
    rm -rf openmpi-4.0.0

ENV JAVA_HOME /usr/lib/jvm/java-1.8-openjdk/jre
ENV PATH $PATH:/usr/lib/jvm/java-1.8-openjdk/jre/bin
ENV MPI_C_COMPILER /usr/local/bin/mpicc
ENV MPI_CXX_COMPILER /usr/local/bin/mpicxx
