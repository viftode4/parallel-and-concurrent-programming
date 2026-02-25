/*
 * Novel Approach: Cache Tiling + Software Prefetching
 *
 * Processes the 3D array in L2-cache-sized tiles (8x8xP = 32K ints = 125 KB).
 * Each tile fits in L2, reducing TLB misses and cache coherence traffic.
 *
 * Key advantage: single-pass min AND max together. The two-section approach
 * (V2/V3) scans the full 477 MB array twice. Tiling with a single pass
 * halves memory bandwidth, which is the primary bottleneck.
 *
 * __builtin_prefetch hints the hardware to preload the next row's data
 * while we're still processing the current row, hiding memory latency.
 *
 * Compile with: gcc -O2 -fopenmp
 */
#include "common.h"

/* Tile sizes tuned for L2 cache (~256 KB = 64K ints)
 * TILE_I * TILE_J * P = 8 * 8 * 500 = 32,000 ints = 125 KB => fits in L2 */
#define TILE_I 8
#define TILE_J 8

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
                    /* Prefetch next row's data while processing current row */
                    if (j + 1 < j_end)
                        __builtin_prefetch(&a[IDX(i, j + 1, 0, N, P)], 0, 1);
                    else if (i + 1 < i_end)
                        __builtin_prefetch(&a[IDX(i + 1, j_start, 0, N, P)], 0, 1);

                    for (int k = 0; k < P; k++) {
                        int val = a[IDX(i, j, k, N, P)];
                        if (val < vmin.val) {
                            vmin.val = val;
                            vmin.i = i;
                            vmin.j = j;
                            vmin.k = k;
                        }
                        if (val > vmax.val) {
                            vmax.val = val;
                            vmax.i = i;
                            vmax.j = j;
                            vmax.k = k;
                        }
                    }
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
