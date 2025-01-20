#include <algorithm>
#include <cstdlib>
#include <utility>

typedef int element_t;

typedef unsigned int length_t;

size_t lcs(element_t* arr_1, element_t* arr_2, size_t len_1, size_t len_2) {
  length_t* mm_ = (length_t*)calloc((len_1 + 1) * 3, sizeof(length_t));

  length_t* mm[3] = {mm_, mm_ + len_1 + 1, mm_ + 2 * (len_1 + 1)};

  size_t diag, mm_index;
  size_t i, j, i_start, i_end;
  length_t *mm_this, *mm_minus_1, *mm_minus_2;

  for (diag = 2, mm_index = 2; diag <= len_1 + len_2;
       ++diag, mm_index = (mm_index + 1) % 3) {
    mm_this = mm[mm_index];
    mm_minus_1 = mm[(mm_index + 2) % 3];
    mm_minus_2 = mm[(mm_index + 1) % 3];

    i_start = diag <= len_2 ? 1 : diag - len_2;
    i_end = std::min(len_1, diag - 1);
#pragma omp parallel for private(i, j)
    for (i = i_start; i <= i_end; ++i) {
      j = diag - i;

      if (arr_1[i - 1] == arr_2[j - 1]) {
        mm_this[i] = mm_minus_2[i - 1] + 1;
      } else {
        mm_this[i] = std::max(mm_minus_1[i], mm_minus_1[i - 1]);
      }
    }
  }

  size_t result = mm[mm_index][len_1];

  free(mm_);

  return result;
}
