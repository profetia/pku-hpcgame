#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

static __always_inline int8_t ring(int8_t a) { return a % 3; }

static __always_inline int find_anchor(int kN, int8_t *A, int j) {
  for (int i = j; i < kN; ++i) {
    if (A[i * kN + j]) {
      return i;
    }
  }

  return -1;
}

static __always_inline void swap_rows(int kN, int8_t *A, int8_t *y, int i,
                                      int j) {
  std::swap(y[i], y[j]);

  for (int k = 0; k < kN; ++k) {
    std::swap(A[i * kN + k], A[j * kN + k]);
  }
}

static __always_inline void solve(int kN, int8_t *A, int8_t *y, int8_t *x) {
  for (int j = 0; j < kN; ++j) {
    int anchor = find_anchor(kN, A, j);

    if (anchor != j) {
      swap_rows(kN, A, y, anchor, j);
    }

    int8_t ajj = ring(A[j * kN + j]);
    for (int k = 0; k < kN; ++k) {
      A[j * kN + k] = ring(A[j * kN + k] * ajj);
    }

    y[j] = ring(y[j] * ajj);
    for (int i = j + 1; i < kN; ++i) {
      int8_t aij = A[i * kN + j];
      for (int k = 0; k < kN; ++k) {
        A[i * kN + k] = ring(A[i * kN + k] - A[j * kN + k] * aij);
      }

      y[i] = ring(y[i] - y[j] * aij);
    }
  }

  for (int i = kN - 1; i >= 0; --i) {
    x[i] = y[i];
    for (int j = i + 1; j < kN; ++j) {
      x[i] = ring(x[i] - A[i * kN + j] * x[j]);
    }
  }
}

int main() {
  int kN2, kN1;
  int *matrix;

  {
    FILE *f = fopen("in.data", "rb");
    fread(&kN2, sizeof(kN2), 1, f);
    fread(&kN1, sizeof(kN1), 1, f);

    matrix = new int[kN2 * kN1];
    fread(matrix, sizeof(int), kN2 * kN1, f);

    fclose(f);
  }

  int *cube = new int[kN2 * kN1]{};
  int kM = 0;
  {
    for (int i = 0; i < kN2; i++) {
      for (int j = 0; j < kN1; j++) {
        if (matrix[i * kN1 + j]) {
          cube[i * kN1 + j] = kM;
          ++kM;
        } else {
          cube[i * kN1 + j] = -1;
        }
      }
    }
  }

  int8_t *A = new int8_t[kM * kM]{};
  int8_t *x = new int8_t[kM]{};
  int8_t *y = new int8_t[kM]{};
  {
    constexpr int DIRECTIONS[5][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {0, 0}};

    for (int i = 0; i < kN2; i++) {
      for (int j = 0; j < kN1; j++) {
        int cj = cube[i * kN1 + j];
        if (cj == -1) {
          continue;
        }

        y[cj] = 3 - matrix[i * kN1 + j];

#pragma unroll
        for (int k = 0; k < 5; k++) {
          int ni = i + DIRECTIONS[k][0];
          int nj = j + DIRECTIONS[k][1];
          if (ni < 0 || ni >= kN2 || nj < 0 || nj >= kN1) {
            continue;
          }

          int ci = cube[ni * kN1 + nj];
          if (ci == -1) {
            continue;
          }

          A[ci * kM + cj] = 1;
        }
      }
    }
  }

  solve(kM, A, y, x);

  {
    memset(matrix, 0, kN2 * kN1 * sizeof(int));
    for (int i = 0; i < kN2; i++) {
      for (int j = 0; j < kN1; j++) {
        int ci = cube[i * kN1 + j];
        if (ci == -1) {
          continue;
        }

        matrix[i * kN1 + j] = x[ci];
      }
    }
  }

  {
    FILE *f = fopen("out.data", "wb");
    fwrite(matrix, sizeof(int), kN2 * kN1, f);
    fclose(f);
  }

  delete[] matrix;
  delete[] cube;
  delete[] A;
  delete[] x;
  delete[] y;

  return 0;
}
