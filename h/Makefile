CXX = g++
NVCC = nvcc
INCLUDE = -I./include
CXXFLAGS = -O3 $(INCLUDE) -std=c++20 -march=core-avx2 -Wall -fopenmp
NVCCFLAGS = -O3 $(INCLUDE) -std=c++20 --expt-relaxed-constexpr --extended-lambda -arch=sm_89
CXXLIBS = 
NVCCLIBS = -lcublas

.PHONY: all clean

all: matdiff baseline matgen matprint driver

matdiff: matdiff.cpp positgemm.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(CXXLIBS)

baseline: baseline.cpp positgemm.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(CXXLIBS)

matgen: matgen.cpp positgemm.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(CXXLIBS)

matprint: matprint.cpp positgemm.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(CXXLIBS)

driver: driver.cpp impl.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(NVCCLIBS)

clean:
	rm -f matdiff matprint baseline matgen driver
