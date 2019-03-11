#!/bin/bash

docker build -t tanyard/im:clr .
#docker login
docker push tanyard/im:clr
