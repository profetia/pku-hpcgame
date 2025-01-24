#include <fstream>
#include <universal/number/posit/posit.hpp>

using fp_t = sw::universal::posit<64, 3>;

struct mat {
    int n, m;
    fp_t *P;
    inline mat() : n(0), m(0), P(nullptr) {}
    mat(const mat &r) = delete;
    mat(mat &&r) = default;
    mat operator=(const mat &r) = delete;
    inline mat(std::ifstream &f) {
        f.read((char *)&n, sizeof(int));
        assert(f);
        f.read((char *)&m, sizeof(int));
        assert(f);
        assert(n > 0);
        assert(m > 0);
        size_t siz = (size_t)n * m;
        assert(siz < (1ull << 60));
        siz *= sizeof(fp_t);
        P = new fp_t[siz];
        f.read((char *)P, siz);
        assert(f);
    }
    inline mat(int n, int m) : n(n), m(m), P(new fp_t[n * m]) {}
    inline ~mat() { delete[] P; }

    inline void dump(std::ofstream &f) {
        f.write((char *)&n, sizeof(int));
        f.write((char *)&m, sizeof(int));
        f.write((char *)P, sizeof(fp_t) * n * m);
        assert(f);
    }
};

inline fp_t abs(fp_t a) { return a > 0 ? a : -a; }

inline std::pair<fp_t, std::pair<int, int>> max_abs_diff(const mat &A, const mat &B) {
    fp_t diff = -1;
    std::pair<int, int> pos = {0, 0};
    assert(A.n == B.n);
    assert(A.m == B.m);
    for (size_t i = 0, siz = A.n * A.m; i < siz; ++i) {
        fp_t d = abs(A.P[i] - B.P[i]);
        if (d > diff) {
            diff = d;
            pos = {i / A.m, i % A.m};
        }
    }
    return {diff, pos};
}

inline mat matmul(const mat &A, const mat &B) {
    assert(A.m == B.n);
    mat C(A.n, B.m);
    for (int i = 0; i < A.n * B.m; ++i)
        C.P[i] = 0;
#pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        for (int k = 0; k < B.n; ++k) {
            for (int j = 0; j < B.m; ++j) {
                C.P[i * C.n + j] += A.P[i * A.n + k] * B.P[k * B.n + j];
            }
        }
    }
    return C;
}
