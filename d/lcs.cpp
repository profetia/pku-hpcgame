#include <arm_neon.h>

#include <algorithm>
#include <cstdlib>
#include <utility>

typedef int element_t;

typedef uint32_t length_t;

constexpr size_t BLOCK_SIZE = 16;

__always_inline int32x4_t vrev128q_s32(int32x4_t v) {
  int32x4_t v_rev = vrev64q_s32(v);
  return vcombine_s32(vget_high_s32(v_rev), vget_low_s32(v_rev));
}

size_t lcs(element_t* arr_1, element_t* arr_2, size_t len_1, size_t len_2) {
  length_t* mm_ = (length_t*)calloc((len_1 + 1) * 3, sizeof(length_t));

  length_t* mm[3] = {mm_, mm_ + len_1 + 1, mm_ + 2 * (len_1 + 1)};

  size_t diag, mm_index;
  size_t i, j, i_start, i_end, i_block_start, i_block_end, i_simd_start,
      i_simd_end, n_simd;
  length_t *mm_this, *mm_minus_1, *mm_minus_2;

  for (diag = 2, mm_index = 2; diag <= len_1 + len_2;
       ++diag, mm_index = (mm_index + 1) % 3) {
    mm_this = mm[mm_index];
    mm_minus_1 = mm[(mm_index + 2) % 3];
    mm_minus_2 = mm[(mm_index + 1) % 3];

    i_start = diag <= len_2 ? 1 : diag - len_2;
    i_end = std::min(len_1, diag - 1);
#pragma omp parallel for private(i, j, i_block_start, i_block_end, \
                                     i_simd_start, i_simd_end, n_simd)
    for (i_block_start = i_start; i_block_start <= i_end;
         i_block_start += BLOCK_SIZE) {
      i_block_end = std::min(i_block_start + BLOCK_SIZE - 1, i_end);

      n_simd = (i_block_end - i_block_start + 1) / 4;
      i_simd_start = i_block_start;
      i_simd_end = i_block_start + n_simd * 4 - 1;

#pragma omp unroll
      for (i = i_simd_start; i <= i_simd_end; i += 4) {
        j = diag - i;

        int32x4_t v_arr_1 = vld1q_s32(arr_1 + i - 1);
        int32x4_t v_arr_2 = vld1q_s32(arr_2 + j - 4);
        v_arr_2 = vrev128q_s32(v_arr_2);

        uint32x4_t v_mm_minus_2 = vld1q_u32(mm_minus_2 + i - 1);
        uint32x4_t v_mm_true_branch = vaddq_u32(v_mm_minus_2, vdupq_n_u32(1));

        uint32x4_t v_mm_minus_1 = vld1q_u32(mm_minus_1 + i - 1);
        uint32x4_t v_mm_minus_1_shift_1 = vld1q_u32(mm_minus_1 + i);
        uint32x4_t v_mm_false_branch =
            vmaxq_u32(v_mm_minus_1, v_mm_minus_1_shift_1);

        uint32x4_t v_mm_if = vceqq_s32(v_arr_1, v_arr_2);
        uint32x4_t v_mm_this =
            vbslq_u32(v_mm_if, v_mm_true_branch, v_mm_false_branch);

        vst1q_u32(mm_this + i, v_mm_this);
      }

      for (i = i_simd_end + 1; i <= i_block_end; ++i) {
        j = diag - i;

        if (arr_1[i - 1] == arr_2[j - 1]) {
          mm_this[i] = mm_minus_2[i - 1] + 1;
        } else {
          mm_this[i] = std::max(mm_minus_1[i], mm_minus_1[i - 1]);
        }
      }
    }
  }

  mm_index = (mm_index + 2) % 3;
  size_t result = mm[mm_index][len_1];

  free(mm_);

  return result;
}
