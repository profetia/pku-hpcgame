#!/bin/bash

expected="7.65468510748275439255e+04" # Intel

cmake -B build
cmake --build build

app="build/program"

time taskset -c 0 $app
