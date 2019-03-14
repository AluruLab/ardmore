#!/bin/bash

docker build -t tanyard/im:pearson .
#docker login
docker push tanyard/im:pearson
