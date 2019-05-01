FROM tanyard/im:base

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/

RUN apt-get install -y \
    openjdk-8-jdk \
    maven \
    && rm -rf /var/cache/apk/*

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python

RUN mkdir -p /m2/repo/

COPY settings.xml /usr/share/maven/conf/settings.xml

ENV PATH /usr/lib/jvm/java-8-openjdk-amd64/bin:/usr/lib/jvm/java-8-openjdk-amd64/jre/bin:$PATH

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
    wget https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz && \
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


