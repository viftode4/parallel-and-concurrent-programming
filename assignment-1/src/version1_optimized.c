/*
 * Version 1 — Optimized: parallel for + declare reduction + contiguous memory + collapse.
 *
 * Improvements over the base version:
 *   1. Contiguous flat array — eliminates 3-level pointer chasing, enables
 *      hardware prefetching, improves cache utilisation.
 *   2. declare reduction(minloc/maxloc) — replaces critical section with an
 *      efficient tree-based reduction. No serialisation during the loop.
 *   3. collapse(2) — flattens i*j = 250,000 iterations for finer-grained
 *      load balancing at high thread counts.
 */
#include "common.h"

int main(void)
{
    int *a;
    int M, N, P;
    read_input_flat(&a, &M, &N, &P);

    MinMaxLoc vmin = { .val = INT_MAX, .i = 0, .j = 0, .k = 0 };
    MinMaxLoc vmax = { .val = INT_MIN, .i = 0, .j = 0, .k = 0 };

    double t_start = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(static) \
            reduction(minloc : vmin) reduction(maxloc : vmax)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                int val = a[IDX(i, j, k, N, P)];
                if (val > vmax.val) {
                    vmax.val = val;
                    vmax.i = i;
                    vmax.j = j;
                    vmax.k = k;
                }
                if (val < vmin.val) {
                    vmin.val = val;
                    vmin.i = i;
                    vmin.j = j;
                    vmin.k = k;
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
