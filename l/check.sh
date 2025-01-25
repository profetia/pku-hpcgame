#!/bin/bash

app="./flute"
input="./data/$1.in.bin"
output="./data/$1.out.bin"

rm -rf output.bin

$app $input output.bin

cmp $output output.bin

if [ $? -eq 0 ]; then
    echo "Ok"
else
    echo "Fail"
fi
