#!/bin/bash

expected="7.65468510748272383353e+04" # Intel

cmake -B build
cmake --build build

app="build/program"

time taskset -c 0 $app
