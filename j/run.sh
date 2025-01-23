#!/usr/bin/bash

module load bisheng/compiler4.1.0/bishengmodule 
module load bisheng/kml2.5.0/kblas/omp

export LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH

make -j

export OMP_NUM_THREADS=2
export BLAS_NUM_THREADS=2

./hpl-ai