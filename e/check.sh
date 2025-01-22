#!/bin/bash

app="./main"
data_dir="./data"

in_dir="/data/e/in"
out_dir="/data/e/out"

input="forest.in"
output="forest.out"

rm -f $in_dir/$input $out_dir/$output

cp $data_dir/$1.in $in_dir/$input

export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core

time mpirun -np 16 $app $in_dir/$input $out_dir/$output

cp $out_dir/$output $output

diff -q $output $data_dir/$1.out
if [ $? -eq 0 ]; then
    echo "Ok"
else
    echo "Fail"
fi
