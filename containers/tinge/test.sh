#!/bin/bash

mpiexec -np 32 /usr/local/bin/tinge-mi -i data/toy-genesXexp.exp -o tinge.adj -p 0.001 -e 0.1
