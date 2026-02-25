/*
 * Version 1: parallel for construct.
 *
 * Data races: In the original loop, all threads would read and write the shared
 * variables min_i, min_j, min_k, max_i, max_j, max_k concurrently. The comparison
 * a[i][j][k] > a[max_i][max_j][max_k] reads max_i/max_j/max_k while another thread
 * may be writing them, causing a check-then-act race and possible torn indices
 * (e.g. max_i from thread A, max_j from thread B).
 *
 * Solution: Each thread keeps private copies of the index variables. After the
 * parallel loop, a critical section merges thread-local results into the shared
 * output. This avoids the need for atomic (which cannot protect the compound
 * multi-variable update).
 *
 * Clauses:
 *   shared(a, M, N, P, g_min_i, ..., g_max_k) — matrix (read-only) and final results
 *   for + schedule(static) — distributes outer loop iterations evenly
 *   critical — serialises the merge of per-thread results
 */
#include "common.h"

int main(void)
{
    int ***a;
    int M, N, P;
    read_input(&a, &M, &N, &P);

    /* Global (shared) result indices */
    int g_min_i = 0, g_min_j = 0, g_min_k = 0;
    int g_max_i = 0, g_max_j = 0, g_max_k = 0;

    double t_start = omp_get_wtime();

    #pragma omp parallel shared(a, M, N, P, g_min_i, g_min_j, g_min_k, g_max_i, g_max_j, g_max_k)
    {
        /* Thread-private index tracking — avoids data races */
        int l_min_i = 0, l_min_j = 0, l_min_k = 0;
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
                    if (a[i][j][k] < a[l_min_i][l_min_j][l_min_k]) {
                        l_min_i = i;
                        l_min_j = j;
                        l_min_k = k;
                    }
                }
            }
        }

        /* Merge thread-local results into global results */
        #pragma omp critical
        {
            if (a[l_max_i][l_max_j][l_max_k] > a[g_max_i][g_max_j][g_max_k]) {
                g_max_i = l_max_i;
                g_max_j = l_max_j;
                g_max_k = l_max_k;
            }
            if (a[l_min_i][l_min_j][l_min_k] < a[g_min_i][g_min_j][g_min_k]) {
                g_min_i = l_min_i;
                g_min_j = l_min_j;
                g_min_k = l_min_k;
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
