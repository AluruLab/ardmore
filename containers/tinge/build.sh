#!/bin/bash

docker build -t tanyard/im:tinge .
#docker login
docker push tanyard/im:tinge
