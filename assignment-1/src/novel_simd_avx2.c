/*
 * Novel Approach: AVX2 SIMD Intrinsics + OpenMP
 *
 * Uses _mm256_cmpgt_epi32 and _mm256_blendv_epi8 to compare 8 ints
 * simultaneously per instruction. Each OpenMP thread processes its chunk
 * with SIMD, then a critical section merges thread-local results.
 *
 * Expected benefit: 3-6x throughput per thread compared to scalar code,
 * because each SIMD instruction operates on 8 elements in one cycle.
 *
 * Compile with: gcc -O2 -mavx2 -fopenmp
 */
#include "common.h"
#include <immintrin.h>

typedef struct { int val; long idx; } ValIdx;

/*
 * SIMD scan of a[lo..hi) for min or max.
 * Processes 8 ints at a time using AVX2, then a scalar tail.
 * Tracks the flat index of the winner.
 */
static ValIdx simd_scan_chunk(const int *a, long lo, long hi, int find_min)
{
    ValIdx result;
    result.val = find_min ? INT_MAX : INT_MIN;
    result.idx = lo;

    long i = lo;
    long len = hi - lo;

    if (len >= 8) {
        __m256i vbest = _mm256_set1_epi32(result.val);
        __m256i vbest_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i vcur_idx  = _mm256_add_epi32(vbest_idx, _mm256_set1_epi32((int)lo));
        __m256i vinc = _mm256_set1_epi32(8);

        long simd_end = lo + (len & ~7L);
        for (i = lo; i < simd_end; i += 8) {
            __m256i vdata = _mm256_loadu_si256((const __m256i *)(a + i));

            __m256i mask;
            if (find_min)
                mask = _mm256_cmpgt_epi32(vbest, vdata);  /* best > data => new min */
            else
                mask = _mm256_cmpgt_epi32(vdata, vbest);  /* data > best => new max */

            vbest     = _mm256_blendv_epi8(vbest, vdata, mask);
            vbest_idx = _mm256_blendv_epi8(vbest_idx, vcur_idx, mask);

            vcur_idx = _mm256_add_epi32(vcur_idx, vinc);
        }

        /* Horizontal reduction across 8 lanes */
        int vals[8], idxs[8];
        _mm256_storeu_si256((__m256i *)vals, vbest);
        _mm256_storeu_si256((__m256i *)idxs, vbest_idx);

        for (int lane = 0; lane < 8; lane++) {
            if (find_min ? (vals[lane] < result.val) : (vals[lane] > result.val)) {
                result.val = vals[lane];
                result.idx = idxs[lane];
            }
        }
    }

    /* Scalar tail */
    for (; i < hi; i++) {
        if (find_min ? (a[i] < result.val) : (a[i] > result.val)) {
            result.val = a[i];
            result.idx = i;
        }
    }

    return result;
}

int main(void)
{
    int *a;
    int M, N, P;
    read_input_flat(&a, &M, &N, &P);
    long total = (long)M * N * P;

    ValIdx gmin = { INT_MAX, 0 };
    ValIdx gmax = { INT_MIN, 0 };

    double t_start = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        long chunk = total / nt;
        long lo = tid * chunk;
        long hi = (tid == nt - 1) ? total : lo + chunk;

        /* Each thread SIMDs through its chunk */
        ValIdx lmin = simd_scan_chunk(a, lo, hi, 1);
        ValIdx lmax = simd_scan_chunk(a, lo, hi, 0);

        #pragma omp critical
        {
            if (lmin.val < gmin.val) gmin = lmin;
            if (lmax.val > gmax.val) gmax = lmax;
        }
    }

    double t_end = omp_get_wtime();

    /* Convert flat index to 3D */
    int mi = (int)(gmin.idx / ((long)N * P));
    int mj = (int)((gmin.idx % ((long)N * P)) / P);
    int mk = (int)(gmin.idx % P);

    int xi = (int)(gmax.idx / ((long)N * P));
    int xj = (int)((gmax.idx % ((long)N * P)) / P);
    int xk = (int)(gmax.idx % P);

    printf("Min = %d at (%d, %d, %d)\n", gmin.val, mi, mj, mk);
    printf("Max = %d at (%d, %d, %d)\n", gmax.val, xi, xj, xk);
    printf("Time: %.6f seconds\n", t_end - t_start);

    free(a);
    return 0;
}
