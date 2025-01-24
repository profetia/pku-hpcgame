SHELL := /bin/bash

CXX := nvcc
CXXFLAGS := -O2 -gencode arch=compute_70,code=sm_70

all: baseline focus

baseline: baseline.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

focus: focus.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

run: focus
	time ./focus

check: run
	python3 check.py data/out_sam.data out.data

clean:
	rm -f baseline focus