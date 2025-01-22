CXX := mpiicpx
CXXFLAGS := -O3 -xHost
all: main baseline

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

baseline: baseline.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f main baseline
