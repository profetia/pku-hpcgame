#include <fstream>
#include "positgemm.h"

int main(int argc, char **argv) {
    assert(argc == 4);
    std::ifstream f1(argv[1], std::ios::binary), f2(argv[2], std::ios::binary);
    std::ofstream out(argv[3], std::ios::binary);
    assert(f1);
    assert(f2);
    assert(out);

    mat A(f1), B(f2);
    mat C = matmul(A, B);

    C.dump(out);
}
