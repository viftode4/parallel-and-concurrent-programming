#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

/* Default matrix dimensions — large enough for measurable parallelism */
#define DEFAULT_M 500
#define DEFAULT_N 500
#define DEFAULT_P 500

/* Random seed for reproducibility */
#define SEED 42

/*
 * Generate a wide-range random int in [0, 99999].
 * MinGW's RAND_MAX is only 32767, so we combine two rand() calls
 * to get enough bits for the full range.
 */
static int wide_rand(void)
{
    return ((rand() << 15) | rand()) % 100000;
}

/*
 * Allocate and fill a 3D matrix a[M][N][P] with random values in [0, 99999].
 * Uses a fixed seed so all versions process identical data.
 *
 * A unique minimum (-1) is planted at (M-1, N-1, P-1) and a unique
 * maximum (100000) at (M/2, N/2, P/2) so correctness can be verified
 * by checking that all versions report the same known indices.
 */
__attribute__((unused))
static void read_input(int ****a, int *M, int *N, int *P)
{
    *M = DEFAULT_M;
    *N = DEFAULT_N;
    *P = DEFAULT_P;

    int m = *M, n = *N, p = *P;

    int ***arr = (int ***)malloc(m * sizeof(int **));
    for (int i = 0; i < m; i++) {
        arr[i] = (int **)malloc(n * sizeof(int *));
        for (int j = 0; j < n; j++) {
            arr[i][j] = (int *)malloc(p * sizeof(int));
        }
    }

    /* Fill with deterministic pseudo-random values in [0, 99999] */
    srand(SEED);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < p; k++)
                arr[i][j][k] = wide_rand();

    /* Plant a guaranteed unique minimum and maximum for verification */
    arr[m - 1][n - 1][p - 1] = -1;       /* unique min */
    arr[m / 2][n / 2][p / 2]  = 100000;  /* unique max */

    *a = arr;
}

/* Free the 3D matrix (pointer-of-pointer version) */
__attribute__((unused))
static void free_matrix(int ***a, int M, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            free(a[i][j]);
        free(a[i]);
    }
    free(a);
}

/* ================================================================
 *  Contiguous (flat) 1D allocation — used by optimized versions.
 *  Eliminates pointer chasing; enables hardware prefetching.
 *  Access element (i,j,k) as: a[i*N*P + j*P + k]
 * ================================================================ */

/* Convenience macro for flat 3D indexing */
#define IDX(i, j, k, N, P) ((i)*(N)*(P) + (j)*(P) + (k))

/*
 * Allocate a contiguous 1D block and fill it identically to read_input()
 * so that both layouts produce the same min/max results.
 */
__attribute__((unused))
static void read_input_flat(int **a, int *M, int *N, int *P)
{
    *M = DEFAULT_M;
    *N = DEFAULT_N;
    *P = DEFAULT_P;

    int m = *M, n = *N, p = *P;

    int *arr = (int *)malloc((size_t)m * n * p * sizeof(int));

    srand(SEED);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < p; k++)
                arr[IDX(i, j, k, n, p)] = wide_rand();

    arr[IDX(m - 1, n - 1, p - 1, n, p)] = -1;       /* unique min */
    arr[IDX(m / 2, n / 2, p / 2, n, p)]  = 100000;   /* unique max */

    *a = arr;
}

/* ================================================================
 *  MinMaxLoc struct + declare reduction — used by optimized V1.
 *  Replaces critical sections with an efficient tree reduction.
 * ================================================================ */

typedef struct {
    int val;
    int i, j, k;
} MinMaxLoc;

/* Combiners (used by declare reduction; may appear unused in some translation units) */
__attribute__((unused))
static void minloc_combine(MinMaxLoc *out, MinMaxLoc *in) {
    if (in->val < out->val) *out = *in;
}
__attribute__((unused))
static void maxloc_combine(MinMaxLoc *out, MinMaxLoc *in) {
    if (in->val > out->val) *out = *in;
}

/* Custom reductions */
#pragma omp declare reduction(minloc : MinMaxLoc : \
        minloc_combine(&omp_out, &omp_in)) \
        initializer(omp_priv = (MinMaxLoc){ .val = INT_MAX, .i = 0, .j = 0, .k = 0 })

#pragma omp declare reduction(maxloc : MinMaxLoc : \
        maxloc_combine(&omp_out, &omp_in)) \
        initializer(omp_priv = (MinMaxLoc){ .val = INT_MIN, .i = 0, .j = 0, .k = 0 })

#endif /* COMMON_H */
