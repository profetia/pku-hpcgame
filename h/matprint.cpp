#include "positgemm.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <bit>

int main(int argc, char **argv) {
    assert(argc == 2);

    std::ifstream f(argv[1]);
    assert(f);
    mat A(f);

    std::cout << "Values" << std::endl;
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.m; ++j) {
            std::cout.width(12);
            std::cout << (double)A.P[i * A.m + j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "Posits" << std::endl;
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.m; ++j) {
            std::cout << sw::universal::pretty_print(A.P[i * A.m + j]) << ' ' << std::hex << std::setw(16) << std::setfill('0') << std::bit_cast<uint64_t>(A.P[i * A.m + j]) << '\n';
        }
        std::cout << '\n';
    }
}