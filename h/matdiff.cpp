#include <fstream>
#include "positgemm.h"

int main(int argc, char **argv) {
    assert(argc == 3);
    std::ifstream f1(argv[1], std::ios::binary);
    std::ifstream f2(argv[2], std::ios::binary);
    assert(f1);
    assert(f2);

    mat A(f1), B(f2);
    auto r = max_abs_diff(A, B);
    std::cout << (double)max_abs_diff(A, B).first << ' ' << '(' << r.second.first << ", " << r.second.second << ')' << std::endl;
}
