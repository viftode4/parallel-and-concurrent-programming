/*
 * Version 3: parallel for + sections (nested parallelism).
 *
 * Outer level: sections split min-finding and max-finding into two tasks.
 * Inner level: within each section, a parallel for distributes the loop
 * iterations across threads.
 *
 * Data races:
 *   Between sections: none — each section works on separate result variables.
 *   Within each section's parallel for: the same race as Version 1 —
 *   multiple inner threads would read/write the index variables concurrently.
 *   Solution: each inner thread uses private local index variables and merges
 *   via a critical section, same pattern as Version 1.
 *
 * Clauses:
 *   sections / section — splits min and max tasks
 *   num_threads(2) — outer level uses exactly 2 threads (one per section)
 *   parallel for + schedule(static) — inner loop parallelism within each section
 *   num_threads(inner_threads) — distributes remaining threads to inner teams
 *   critical — protects the merge of inner thread-local results
 */
#include "common.h"

int main(void)
{
    int ***a;
    int M, N, P;
    read_input(&a, &M, &N, &P);

    int g_min_i = 0, g_min_j = 0, g_min_k = 0;
    int g_max_i = 0, g_max_j = 0, g_max_k = 0;

    /* Enable nested parallelism */
    omp_set_nested(1);
    omp_set_max_active_levels(2);

    int total_threads = omp_get_max_threads();
    int inner_threads = (total_threads > 2) ? total_threads / 2 : 1;

    double t_start = omp_get_wtime();

    #pragma omp parallel sections shared(a, M, N, P, g_min_i, g_min_j, g_min_k, g_max_i, g_max_j, g_max_k) num_threads(2)
    {
        /* Section 1: find minimum using parallel for */
        #pragma omp section
        {
            #pragma omp parallel shared(a, M, N, P, g_min_i, g_min_j, g_min_k) num_threads(inner_threads)
            {
                int l_min_i = 0, l_min_j = 0, l_min_k = 0;

                #pragma omp for schedule(static)
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        for (int k = 0; k < P; k++) {
                            if (a[i][j][k] < a[l_min_i][l_min_j][l_min_k]) {
                                l_min_i = i;
                                l_min_j = j;
                                l_min_k = k;
                            }
                        }
                    }
                }

                #pragma omp critical(min_merge)
                {
                    if (a[l_min_i][l_min_j][l_min_k] < a[g_min_i][g_min_j][g_min_k]) {
                        g_min_i = l_min_i;
                        g_min_j = l_min_j;
                        g_min_k = l_min_k;
                    }
                }
            }
        }

        /* Section 2: find maximum using parallel for */
        #pragma omp section
        {
            #pragma omp parallel shared(a, M, N, P, g_max_i, g_max_j, g_max_k) num_threads(inner_threads)
            {
                int l_max_i = 0, l_max_j = 0, l_max_k = 0;

                #pragma omp for schedule(static)
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        for (int k = 0; k < P; k++) {
                            if (a[i][j][k] > a[l_max_i][l_max_j][l_max_k]) {
                                l_max_i = i;
                                l_max_j = j;
                                l_max_k = k;
                            }
                        }
                    }
                }

                #pragma omp critical(max_merge)
                {
                    if (a[l_max_i][l_max_j][l_max_k] > a[g_max_i][g_max_j][g_max_k]) {
                        g_max_i = l_max_i;
                        g_max_j = l_max_j;
                        g_max_k = l_max_k;
                    }
                }
            }
        }
    }

    double t_end = omp_get_wtime();

    printf("Min = %d at (%d, %d, %d)\n", a[g_min_i][g_min_j][g_min_k], g_min_i, g_min_j, g_min_k);
    printf("Max = %d at (%d, %d, %d)\n", a[g_max_i][g_max_j][g_max_k], g_max_i, g_max_j, g_max_k);
    printf("Time: %.6f seconds\n", t_end - t_start);

    free_matrix(a, M, N);
    return 0;
}
