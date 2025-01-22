CXX = g++
CFLAGS = -O3 -march=native
LIBS = -lsecp256k1 -lcrypto -fopenmp -lpthread

.PHONY: all clean

all: vanity baseline chk

vanity: vanity.cpp
	$(CXX) $(CFLAGS) -o $@ $< $(LIBS)

baseline: baseline.cpp
	$(CXX) $(CFLAGS) -o $@ $< $(LIBS)

chk: chk.cpp
	$(CXX) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f vanity chk