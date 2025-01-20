#!/bin/bash

data_dir="data"
app="./lcs"

cores=$(cat /proc/self/status | grep "Cpus_allowed_list" | cut -f2)
printf "Benchmarking with $cores\n"

function check_impl {
    echo "Checking case $1"
    rm -f input.dat output.txt
    cp $data_dir/case$1.in input.dat
    OMP_NUM_THREADS=16 taskset -c $cores $app
    diff -q $data_dir/case$1.out output.txt
    if [ $? -eq 0 ]; then
        echo "$1: Ok"
    else
        echo "$1: Fail"
    fi
}

if [ $# -eq 1 ]; then
    check_impl $1
    exit
fi

for i in {0..4}; do
    check_impl $i
done