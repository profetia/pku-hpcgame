#!/bin/bash

app="./baseline"
data="./data"

input="./forest.in"
output="./forest.out"

rm -f $input $output

cp $data/$1.in $input

export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core

time $app $input $output

diff -q $output $data/$1.out
if [ $? -eq 0 ]; then
    echo "Ok"
else
    echo "Fail"
fi
