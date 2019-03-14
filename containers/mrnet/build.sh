#!/bin/bash

docker build -t tanyard/im:mrnet .
#docker login
docker push tanyard/im:mrnet
