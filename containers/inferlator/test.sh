#!/bin/bash
singularity pull docker://tanyard/im:inferlator
singularity run im_inferlator.sif Rscript /usr/local/bin/inferelator.R config.R
