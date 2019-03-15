#!/bin/bash

docker build -t tanyard/im:inferlator .
#docker login
docker push tanyard/im:inferlator
