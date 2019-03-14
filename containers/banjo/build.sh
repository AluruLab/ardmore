#!/bin/bash

docker build -t tanyard/im:banjo .
#docker login
docker push tanyard/im:banjo
