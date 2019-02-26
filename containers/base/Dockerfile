FROM alpine:3.9

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
    && rm -rf /var/cache/apk/*

# make sure pip is available
RUN python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
    if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 /usr/bin/python; fi && \
    rm -r /root/.cache

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/jre/
ENV PATH $PATH:/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/
