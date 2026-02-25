/*
 * Novel Approach: OpenMP SIMD Directive — Two-Pass
 *
 * Pass 1: #pragma omp parallel for simd reduction(min:vmin) reduction(max:vmax)
 *   Finds the min/max VALUES only. GCC auto-vectorizes this to use SSE/AVX
 *   instructions (vpminsd/vpmaxsd), processing 4-8 ints per cycle.
 *
 * Pass 2: #pragma omp parallel for schedule(static)
 *   Finds the INDICES of the already-known min/max values with a simple scan.
 *   This pass is over cache-hot data so the cost is low.
 *
 * Why two passes? GCC cannot vectorize struct-based reductions (MinMaxLoc)
 * because the struct access pattern is not representable as SIMD lanes.
 * Splitting into value-only (vectorizable) + index-finding (scalar) is faster
 * than a single non-vectorized pass.
 *
 * Compile with: gcc -O2 -fopenmp -mavx2  (or just -O2 -fopenmp for SSE4)
 */
#include "common.h"

int main(void)
{
    int *a;
    int M, N, P;
    read_input_flat(&a, &M, &N, &P);
    long total = (long)M * N * P;

    int vmin = INT_MAX;
    int vmax = INT_MIN;
    long min_idx = 0, max_idx = 0;

    double t_start = omp_get_wtime();

    /* ---- Pass 1: Find VALUES (fully auto-vectorized) ---- */
    #pragma omp parallel for simd reduction(min:vmin) reduction(max:vmax) \
            schedule(static) simdlen(8)
    for (long i = 0; i < total; i++) {
        if (a[i] < vmin) vmin = a[i];
        if (a[i] > vmax) vmax = a[i];
    }

    /* ---- Pass 2: Find INDICES (data is cache-hot from pass 1) ---- */
    int found_min = 0, found_max = 0;

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < total; i++) {
        if (!found_min && a[i] == vmin) {
            #pragma omp critical(idx_min)
            {
                if (!found_min) {
                    min_idx = i;
                    found_min = 1;
                }
            }
        }
        if (!found_max && a[i] == vmax) {
            #pragma omp critical(idx_max)
            {
                if (!found_max) {
                    max_idx = i;
                    found_max = 1;
                }
            }
        }
    }

    double t_end = omp_get_wtime();

    /* Convert flat index to 3D */
    int mi = (int)(min_idx / ((long)N * P));
    int mj = (int)((min_idx % ((long)N * P)) / P);
    int mk = (int)(min_idx % P);

    int xi = (int)(max_idx / ((long)N * P));
    int xj = (int)((max_idx % ((long)N * P)) / P);
    int xk = (int)(max_idx % P);

    printf("Min = %d at (%d, %d, %d)\n", vmin, mi, mj, mk);
    printf("Max = %d at (%d, %d, %d)\n", vmax, xi, xj, xk);
    printf("Time: %.6f seconds\n", t_end - t_start);

    free(a);
    return 0;
}
