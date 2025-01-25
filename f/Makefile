CXX := g++
CXXFLAGS := -O3 -march=native -fopenmp

all: main

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f main