CXX := clang++
CXXFLAGS := -O3 -march=native

all: baseline flute datagen

datagen: datagen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

baseline: baseline.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

flute: flute.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f baseline flute datagen