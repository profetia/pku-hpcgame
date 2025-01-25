#!/bin/bash

# app="./main"
app="sage -python baseline.py"
input="./data/$1.in.data"
answer="./data/$1.out.data"

rm -f ./in.data ./out.data
cp $input ./in.data
$app
diff -q ./out.data $answer
if [ $? -eq 0 ]; then
    echo "Ok"
else
    echo "Fail"
fi