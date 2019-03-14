#!/bin/bash

docker build -t tanyard/im:genenet .
#docker login
docker push tanyard/im:genenet
