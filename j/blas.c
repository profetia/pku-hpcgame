#include <kblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hpl-ai.h"

#define A(i, j) *HPLAI_INDEX2D(A, (i), (j), lda)
#define B(i, j) *HPLAI_INDEX2D(B, (i), (j), ldb)
#define C(i, j) *HPLAI_INDEX2D(C, (i), (j), ldc)

void sgemm(char transa, char transb, int m, int n, int k, float alpha, float *A,
           int lda, float *B, int ldb, float beta, float *C, int ldc) {
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda,
              B, ldb, beta, C, ldc);
}

void strsm(char side, char uplo, char transa, char diag, int m, int n,
           float alpha, float *A, int lda, float *B, int ldb) {
  if (uplo == 'U') {
    if (diag == 'U') {
      cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit,
                  m, n, alpha, A, lda, B, ldb);
    } else {
      cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                  CblasNonUnit, m, n, alpha, A, lda, B, ldb);
    }
  } else {
    if (diag == 'U') {
      cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                  m, n, alpha, A, lda, B, ldb);
    } else {
      cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                  CblasNonUnit, m, n, alpha, A, lda, B, ldb);
    }
  }
}

void dtrsm(char side, char uplo, char transa, char diag, int m, int n,
           double alpha, double *A, int lda, double *B, int ldb) {
  if (uplo == 'U') {
    if (diag == 'U') {
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit,
                  m, n, alpha, A, lda, B, ldb);
    } else {
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                  CblasNonUnit, m, n, alpha, A, lda, B, ldb);
    }
  } else {
    if (diag == 'U') {
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                  m, n, alpha, A, lda, B, ldb);
    } else {
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                  CblasNonUnit, m, n, alpha, A, lda, B, ldb);
    }
  }
}

double dlange(char norm, int m, int n, double *A, int lda) {
  int i, j;

  // Frobenius norm
  if (norm == 'F') {
    double sum = 0.0;
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) {
        sum += A(i, j) * A(i, j);
      }
    }
    return sqrt(sum);
    // Infinity norm
  } else if (norm == 'I') {
    double *work = (double *)malloc(m * sizeof(double));
    memset(work, 0, m * sizeof(double));
    double max = 0.0;
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) {
        work[i] += fabs(A(i, j));
      }
    }
    for (i = 0; i < m; ++i) {
      if (max < work[i]) {
        max = work[i];
      }
    }
    free(work);
    return max;
  }
  return 0;
}

void dgemv(char trans, int m, int n, double alpha, double *A, int lda,
           double *X, int incx, double beta, double *Y, int incy) {
  cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, A, lda, X, incx, beta,
              Y, incy);
}
