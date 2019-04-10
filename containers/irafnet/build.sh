#!/bin/bash

docker build -t tanyard/im:irafnet .
#docker login
docker push tanyard/im:irafnet
