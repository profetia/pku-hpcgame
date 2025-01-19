CXX=clang++
CXXFLAGS=-O2 -std=c++20 -fopenmp -mcpu=hip09 
LDFLAGS=-fopenmp

.PHONY: all clean

all: lcs generate

lcs: main.cpp lcs.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

generate: generate.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

data: generate
	./scripts/data.sh

bless: lcs
	./scripts/bless.sh

check: lcs
	./scripts/check.sh

benchmark: lcs
	./scripts/benchmark.sh

clean:
	rm -f lcs generate