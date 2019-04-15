FROM tanyard/im:base

# # taken from github.com/jeanblanchard/docker-alpine-glibc  to fix the glibc issue
# ENV GLIBC_VERSION 2.29-r0
# # Download and install glibc
# RUN apk add --update curl && \
#   curl -Lo /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub && \
#   curl -Lo glibc.apk "https://github.com/sgerrand/alpine-pkg-glibc/releases/download/${GLIBC_VERSION}/glibc-${GLIBC_VERSION}.apk" && \
#   curl -Lo glibc-bin.apk "https://github.com/sgerrand/alpine-pkg-glibc/releases/download/${GLIBC_VERSION}/glibc-bin-${GLIBC_VERSION}.apk" && \
#   apk add glibc-bin.apk glibc.apk && \
#   /usr/glibc-compat/sbin/ldconfig /lib /usr/glibc-compat/lib && \
#   echo 'hosts: files mdns4_minimal [NOTFOUND=return] dns mdns4' >> /etc/nsswitch.conf && \
#   apk del curl && \
# rm -rf glibc.apk glibc-bin.apk /var/cache/apk/*

RUN mkdir -p /work/ && cd /work && \
    wget http://m3d.mssm.edu/network_inference/clr_v1.2.2.tar.gz && \
    tar -xf clr_v1.2.2.tar.gz && \
    cd CLRv1.2.2/Code 

COPY Makefile /work/CLRv1.2.2/Code/InfoKit2/ 

RUN cd /work/CLRv1.2.2/Code/InfoKit2/ &&  \
    make linux64 && \
    cp clr /usr/local/bin
