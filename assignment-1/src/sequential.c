/*
 * Sequential baseline — direct translation of the given pseudocode.
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

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                if (a[i][j][k] > a[max_i][max_j][max_k]) {
                    max_i = i;
                    max_j = j;
                    max_k = k;
                }
                if (a[i][j][k] < a[min_i][min_j][min_k]) {
                    min_i = i;
                    min_j = j;
                    min_k = k;
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
