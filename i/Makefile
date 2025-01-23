CXX := nvcc
CXXFLAGS := -O2 -gencode arch=compute_80,code=sm_80

all: baseline focus

baseline: baseline.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

focus: focus.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f baseline focus