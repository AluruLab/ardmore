#!/bin/bash

docker build -t tanyard/im:grnboost .
#docker login
docker push tanyard/im:grnboost
