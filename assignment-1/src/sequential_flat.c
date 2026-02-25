/*
 * Sequential baseline using contiguous (flat) memory layout.
 * This is the fair comparison baseline for the optimized parallel versions.
 */
#include "common.h"

int main(void)
{
    int *a;
    int M, N, P;
    read_input_flat(&a, &M, &N, &P);

    int min_i = 0, min_j = 0, min_k = 0;
    int max_i = 0, max_j = 0, max_k = 0;

    double t_start = omp_get_wtime();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                if (a[IDX(i, j, k, N, P)] > a[IDX(max_i, max_j, max_k, N, P)]) {
                    max_i = i;
                    max_j = j;
                    max_k = k;
                }
                if (a[IDX(i, j, k, N, P)] < a[IDX(min_i, min_j, min_k, N, P)]) {
                    min_i = i;
                    min_j = j;
                    min_k = k;
                }
            }
        }
    }

    double t_end = omp_get_wtime();

    printf("Min = %d at (%d, %d, %d)\n", a[IDX(min_i, min_j, min_k, N, P)], min_i, min_j, min_k);
    printf("Max = %d at (%d, %d, %d)\n", a[IDX(max_i, max_j, max_k, N, P)], max_i, max_j, max_k);
    printf("Time: %.6f seconds\n", t_end - t_start);

    free(a);
    return 0;
}
