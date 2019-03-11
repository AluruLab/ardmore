#!/bin/bash

docker build -t tanyard/im:aracne .
#docker login
docker push tanyard/im:aracne
