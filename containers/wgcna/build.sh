#!/bin/bash

docker build -t tanyard/im:wgcna .
#docker login
docker push tanyard/im:wgcna
