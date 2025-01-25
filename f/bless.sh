#!/bin/bash

baseline="sage -python baseline.py"
input="./data/$1.in.data"
output="./data/$1.out.data"

rm -f ./in.data ./out.data
cp $input ./in.data
$baseline
cp ./out.data $output
