#!/bin/bash

./matgen matA.bin $1 $1
./matgen matB.bin $1 $1
./driver matA.bin matB.bin matC.bin
./baseline matA.bin matB.bin matCstd.bin
./matdiff matC.bin matCstd.bin
