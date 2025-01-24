#include <fstream>
#include <random>
#include "positgemm.h"

int main(int argc, char **argv) {
    assert(argc == 4);
    std::ofstream out(argv[1]);
    assert(out);
    
    int n = atoi(argv[2]), m = atoi(argv[3]);
    assert(n > 0);
    assert(m > 0);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(-(2 - 1e-4), 2 - 1e-4);
    mat A(n, m);
    for (int i = 0; i < n * m; ++i) {
        A.P[i] = dist(rng);
    }
    A.dump(out);
}