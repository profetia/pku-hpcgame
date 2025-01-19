#!/bin/bash

data_dir="data"
app="./lcs"

for i in {0..4}; do
    echo "Checking case $i"

    rm -f input.dat output.txt
    cp $data_dir/case$i.in input.dat
    $app
    diff -q $data_dir/case$i.out output.txt
    if [ $? -eq 0 ]; then
        echo "$i: Ok"
    else
        echo "$i: Fail"
    fi
done