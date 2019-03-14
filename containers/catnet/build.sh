#!/bin/bash

docker build -t tanyard/im:catnet .
#docker login
docker push tanyard/im:catnet
