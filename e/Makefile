CXX := mpiicpx
CXXFLAGS := -O3 -xHost -Wall -Wpedantic
all: main baseline

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

baseline: baseline.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f main baseline
