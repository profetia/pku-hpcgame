#!/bin/bash

data_dir="data"
mkdir -p $data_dir

cases=(
    "65536 65536" 
    "65536 262144" 
    "262144 262144" 
    "262144 1048576" 
    "1048576 1048576"
)

for i in {0..4}; do
    echo "Generating case $i"
    ./generate ${cases[$i]} $data_dir/case$i.in
done
