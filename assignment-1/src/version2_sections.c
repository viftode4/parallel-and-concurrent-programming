/*
 * Version 2: sections construct.
 *
 * Data races: We split the work so that one section finds the minimum and
 * another finds the maximum. Since each section writes to completely separate
 * variables (min_i/j/k vs max_i/j/k), there are no data races between
 * sections. The matrix a[] is read-only, which is also safe.
 * Maximum parallelism is limited to 2 threads (one per section).
 *
 * Clauses:
 *   shared(a, M, N, P, min_i, ..., max_k) — matrix and results visible to both sections
 *   sections / section — assigns each section to a different thread
 */
#include "common.h"

int main(void)
{
    int ***a;
    int M, N, P;
    read_input(&a, &M, &N, &P);

    int min_i = 0, min_j = 0, min_k = 0;
    int max_i = 0, max_j = 0, max_k = 0;

    double t_start = omp_get_wtime();

    #pragma omp parallel sections shared(a, M, N, P, min_i, min_j, min_k, max_i, max_j, max_k)
    {
        /* Section 1: find minimum */
        #pragma omp section
        {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < P; k++) {
                        if (a[i][j][k] < a[min_i][min_j][min_k]) {
                            min_i = i;
                            min_j = j;
                            min_k = k;
                        }
                    }
                }
            }
        }

        /* Section 2: find maximum */
        #pragma omp section
        {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < P; k++) {
                        if (a[i][j][k] > a[max_i][max_j][max_k]) {
                            max_i = i;
                            max_j = j;
                            max_k = k;
                        }
                    }
                }
            }
        }
    }

    double t_end = omp_get_wtime();

    printf("Min = %d at (%d, %d, %d)\n", a[min_i][min_j][min_k], min_i, min_j, min_k);
    printf("Max = %d at (%d, %d, %d)\n", a[max_i][max_j][max_k], max_i, max_j, max_k);
    printf("Time: %.6f seconds\n", t_end - t_start);

    free_matrix(a, M, N);
    return 0;
}
