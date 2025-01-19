#!/bin/bash

data_dir="data"
app="./lcs"

cores=$(cat /proc/self/status | grep "Cpus_allowed_list" | cut -f2)
printf "Benchmarking with $cores\n"

for i in {0..4}; do
    echo "Benchmarking case $i"

    rm -f input.dat output.txt
    cp $data_dir/case$i.in input.dat
    time taskset -c $cores $app
done