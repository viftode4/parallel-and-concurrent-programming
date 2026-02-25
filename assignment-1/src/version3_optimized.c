/*
 * Version 3 — Optimized: sections + parallel for + contiguous memory +
 *             declare reduction + collapse.
 *
 * Improvements over the base version:
 *   1. Contiguous flat array — same cache/prefetch benefits as V1 optimized.
 *   2. declare reduction — each inner parallel for uses a tree reduction
 *      instead of a critical section.
 *   3. collapse(2) — finer-grained distribution within each section.
 */
#include "common.h"

int main(void)
{
    int *a;
    int M, N, P;
    read_input_flat(&a, &M, &N, &P);

    MinMaxLoc vmin = { .val = INT_MAX, .i = 0, .j = 0, .k = 0 };
    MinMaxLoc vmax = { .val = INT_MIN, .i = 0, .j = 0, .k = 0 };

    omp_set_nested(1);
    omp_set_max_active_levels(2);

    int total_threads = omp_get_max_threads();
    int inner_threads = (total_threads > 2) ? total_threads / 2 : 1;

    double t_start = omp_get_wtime();

    #pragma omp parallel sections num_threads(2)
    {
        /* Section 1: find minimum */
        #pragma omp section
        {
            #pragma omp parallel for collapse(2) schedule(static) \
                    reduction(minloc : vmin) num_threads(inner_threads)
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < P; k++) {
                        int val = a[IDX(i, j, k, N, P)];
                        if (val < vmin.val) {
                            vmin.val = val;
                            vmin.i = i;
                            vmin.j = j;
                            vmin.k = k;
                        }
                    }
                }
            }
        }

        /* Section 2: find maximum */
        #pragma omp section
        {
            #pragma omp parallel for collapse(2) schedule(static) \
                    reduction(maxloc : vmax) num_threads(inner_threads)
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
