/*
 * Novel Approach: Ultimate — AVX2 SIMD + Cache Tiling + Prefetching
 *
 * Combines the two best-performing techniques:
 *   - Cache tiling (8x8xP tiles fit in L2) for optimal memory access
 *   - AVX2 SIMD intrinsics (8 ints/instruction) for maximum compute throughput
 *   - Software prefetching to hide memory latency between rows
 *   - declare reduction for efficient tree-based merge
 *   - collapse(2) on tile loops for fine-grained work distribution
 *   - Single pass: finds both min AND max together
 *
 * This addresses all three bottlenecks simultaneously:
 *   COMPUTE → SIMD processes 8 elements per cycle
 *   BANDWIDTH → single-pass halves memory traffic vs two-section approaches
 *   LATENCY → tiling + prefetch keep data in L2 and hide stalls
 *
 * Compile with: gcc -O2 -mavx2 -fopenmp
 */
#include "common.h"
#include <immintrin.h>

#define TILE_I 8
#define TILE_J 8

/*
 * SIMD scan of one row a[base..base+len) for both min and max.
 * Updates MinMaxLoc structs with the 3D indices (i, j, k).
 */
static inline void simd_scan_row(const int *a, int base, int len,
                                  MinMaxLoc *vmin, MinMaxLoc *vmax,
                                  int row_i, int row_j)
{
    int k = 0;

    if (len >= 8) {
        __m256i vmin_val = _mm256_set1_epi32(vmin->val);
        __m256i vmax_val = _mm256_set1_epi32(vmax->val);
        __m256i vmin_k   = _mm256_set1_epi32(vmin->k);
        __m256i vmax_k   = _mm256_set1_epi32(vmax->k);
        __m256i vcur_k   = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i vinc     = _mm256_set1_epi32(8);

        int simd_end = len & ~7;
        for (k = 0; k < simd_end; k += 8) {
            __m256i vdata = _mm256_loadu_si256((const __m256i *)(a + base + k));

            /* Min update */
            __m256i min_mask = _mm256_cmpgt_epi32(vmin_val, vdata);
            vmin_val = _mm256_blendv_epi8(vmin_val, vdata, min_mask);
            vmin_k   = _mm256_blendv_epi8(vmin_k, vcur_k, min_mask);

            /* Max update */
            __m256i max_mask = _mm256_cmpgt_epi32(vdata, vmax_val);
            vmax_val = _mm256_blendv_epi8(vmax_val, vdata, max_mask);
            vmax_k   = _mm256_blendv_epi8(vmax_k, vcur_k, max_mask);

            vcur_k = _mm256_add_epi32(vcur_k, vinc);
        }

        /* Horizontal reduction across 8 lanes */
        int min_vals[8], min_ks[8], max_vals[8], max_ks[8];
        _mm256_storeu_si256((__m256i *)min_vals, vmin_val);
        _mm256_storeu_si256((__m256i *)min_ks,   vmin_k);
        _mm256_storeu_si256((__m256i *)max_vals, vmax_val);
        _mm256_storeu_si256((__m256i *)max_ks,   vmax_k);

        for (int lane = 0; lane < 8; lane++) {
            if (min_vals[lane] < vmin->val) {
                vmin->val = min_vals[lane];
                vmin->i = row_i; vmin->j = row_j; vmin->k = min_ks[lane];
            }
            if (max_vals[lane] > vmax->val) {
                vmax->val = max_vals[lane];
                vmax->i = row_i; vmax->j = row_j; vmax->k = max_ks[lane];
            }
        }
    }

    /* Scalar tail */
    for (; k < len; k++) {
        int val = a[base + k];
        if (val < vmin->val) {
            vmin->val = val; vmin->i = row_i; vmin->j = row_j; vmin->k = k;
        }
        if (val > vmax->val) {
            vmax->val = val; vmax->i = row_i; vmax->j = row_j; vmax->k = k;
        }
    }
}

int main(void)
{
    int *a;
    int M, N, P;
    read_input_flat(&a, &M, &N, &P);

    MinMaxLoc vmin = { .val = INT_MAX, .i = 0, .j = 0, .k = 0 };
    MinMaxLoc vmax = { .val = INT_MIN, .i = 0, .j = 0, .k = 0 };

    int ni_tiles = (M + TILE_I - 1) / TILE_I;
    int nj_tiles = (N + TILE_J - 1) / TILE_J;

    double t_start = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(static) \
            reduction(minloc : vmin) reduction(maxloc : vmax)
    for (int ti = 0; ti < ni_tiles; ti++) {
        for (int tj = 0; tj < nj_tiles; tj++) {
            int i_start = ti * TILE_I;
            int i_end   = (i_start + TILE_I < M) ? i_start + TILE_I : M;
            int j_start = tj * TILE_J;
            int j_end   = (j_start + TILE_J < N) ? j_start + TILE_J : N;

            for (int i = i_start; i < i_end; i++) {
                for (int j = j_start; j < j_end; j++) {
                    /* Prefetch next row */
                    if (j + 1 < j_end)
                        __builtin_prefetch(&a[IDX(i, j + 1, 0, N, P)], 0, 1);
                    else if (i + 1 < i_end)
                        __builtin_prefetch(&a[IDX(i + 1, j_start, 0, N, P)], 0, 1);

                    /* SIMD scan this row for both min and max */
                    simd_scan_row(a, IDX(i, j, 0, N, P), P,
                                  &vmin, &vmax, i, j);
                }
            }
        }
    }

    double t_end = omp_get_wtime();

    printf("Min = %d at (%d, %d, %d)\n", vmin.val, vmin.i, vmin.j, vmin.k);
    printf("Max = %d at (%d, %d, %d)\n", vmax.val, vmax.i, vmax.j, vmax.k);
    printf("Time: %.6f seconds\n", t_end - t_start);

    free(a);
    return 0;
}
