#!/bin/bash

data_dir="data"
app="./lcs"

cores=$(cat /proc/self/status | grep "Cpus_allowed_list" | cut -f2)
printf "Benchmarking with $cores\n"

function benmark_impl {
    echo "Benchmarking case $1"
    rm -f input.dat output.txt
    cp $data_dir/case$1.in input.dat
    time taskset -c $cores $app
}

if [ $# -eq 1 ]; then
    benmark_impl $1
    exit
fi

for i in {0..4}; do
    benmark_impl $i
done

# baseline: 
# 0: 11.964387
# 1: 47.845859
# 2: 191.401608
# 3: 764.414295
# 4: 3056.486387
