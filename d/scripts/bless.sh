#!/bin/bash

data_dir="data"
app="./lcs"

for i in {0..4}; do
    echo "Blessing case $i"

    rm -f input.dat output.txt
    cp $data_dir/case$i.in input.dat
    $app
    cp output.txt $data_dir/case$i.out
done
