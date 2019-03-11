#!/bin/bash

docker build -t tanyard/im:fastggm .
#docker login
docker push tanyard/im:fastggm
