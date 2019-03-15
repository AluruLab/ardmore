FROM tanyard/im:base

ENV JAVA_HOME /usr/lib/jvm/java-1.8-openjdk/

RUN apk add --update \
    bash \
    maven \
    && rm -rf /var/cache/apk/*

RUN mkdir -p /m2/repo/

COPY settings.xml /usr/share/java/maven-3/conf/settings.xml

RUN mkdir -p /work/ && cd /work && \
    git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4 && \
    make install && \
    cd ../jvm-packages && \
    mvn -DskipTests install

RUN cp /work/xgboost/lib/*.so /usr/local/lib/ 

RUN mkdir -p /work/ && cd /work && \
    wget http://mirrors.ocf.berkeley.edu/apache/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz && \
    tar xvzf spark-2.4.0-bin-hadoop2.7.tgz && \
    mv spark-2.4.0-bin-hadoop2.7/ /usr/local/bin/spark/ && \
    rm -rf spark-2.4.0-bin-hadoop2.7.tgz

RUN mkdir -p /work/ && cd /work && \
    wget https://piccolo.link/sbt-1.2.8.tgz && \
    tar xvzf sbt-1.2.8.tgz && \
    mv sbt /usr/local/bin/ && \
    rm -rf sbt-1.2.8.tgz


RUN mkdir -p /work/ && cd /work && \
    git clone https://github.com/aertslab/GRNBoost/ && \
    cd GRNBoost && \
    /usr/local/bin/sbt/bin/sbt assembly


