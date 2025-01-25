CXX := nvcc
CXXFLAGS := -O3 -gencode=arch=compute_89,code=sm_89 -Xcompiler -fopenmp

all: baseline flute datagen

datagen: datagen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

baseline: baseline.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

flute: flute.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f baseline flute datagen