/*
 * Novel Approach: Branchless Bitwise Min/Max
 *
 * Uses XOR-based conditional selection to update min/max and their indices
 * without any branch instructions. The key trick:
 *   mask = -(condition)   // 0x00000000 or 0xFFFFFFFF
 *   result = old ^ ((old ^ new) & mask)
 *
 * This avoids branch misprediction penalties entirely. On random data the
 * branch predictor is ~99.99% accurate (updates are rare), so the benefit
 * is small. On adversarial data (sorted descending), every element triggers
 * an update and branchless wins by avoiding ~10-20 cycle misprediction
 * penalties per element.
 *
 * The 'nowait' clause on the for loop avoids a redundant barrier before the
 * critical section, since each thread can merge immediately when done.
 *
 * Compile with: gcc -O2 -fopenmp
 */
#include "common.h"

/* Branchless conditional select: returns a if cond != 0, else b.
 * Pure arithmetic — no branch instructions generated. */
static inline int select_int(int cond, int a, int b)
{
    int mask = -cond;  /* 0 -> 0x00000000, 1 -> 0xFFFFFFFF */
    return (a & mask) | (b & ~mask);
}

static inline long select_long(int cond, long a, long b)
{
    long mask = -(long)cond;
    return (a & mask) | (b & ~mask);
}

int main(void)
{
    int *a;
    int M, N, P;
    read_input_flat(&a, &M, &N, &P);
    long total = (long)M * N * P;

    int gmin_val = INT_MAX, gmax_val = INT_MIN;
    long gmin_idx = 0, gmax_idx = 0;

    double t_start = omp_get_wtime();

    #pragma omp parallel
    {
        int lmin_val = INT_MAX, lmax_val = INT_MIN;
        long lmin_idx = 0, lmax_idx = 0;

        #pragma omp for schedule(static) nowait
        for (long i = 0; i < total; i++) {
            int val = a[i];

            /* Branchless min update */
            int is_less = (val < lmin_val);
            lmin_val = select_int(is_less, val, lmin_val);
            lmin_idx = select_long(is_less, i, lmin_idx);

            /* Branchless max update */
            int is_greater = (val > lmax_val);
            lmax_val = select_int(is_greater, val, lmax_val);
            lmax_idx = select_long(is_greater, i, lmax_idx);
        }

        #pragma omp critical
        {
            if (lmin_val < gmin_val) { gmin_val = lmin_val; gmin_idx = lmin_idx; }
            if (lmax_val > gmax_val) { gmax_val = lmax_val; gmax_idx = lmax_idx; }
        }
    }

    double t_end = omp_get_wtime();

    /* Convert flat index to 3D */
    int mi = (int)(gmin_idx / ((long)N * P));
    int mj = (int)((gmin_idx % ((long)N * P)) / P);
    int mk = (int)(gmin_idx % P);

    int xi = (int)(gmax_idx / ((long)N * P));
    int xj = (int)((gmax_idx % ((long)N * P)) / P);
    int xk = (int)(gmax_idx % P);

    printf("Min = %d at (%d, %d, %d)\n", gmin_val, mi, mj, mk);
    printf("Max = %d at (%d, %d, %d)\n", gmax_val, xi, xj, xk);
    printf("Time: %.6f seconds\n", t_end - t_start);

    free(a);
    return 0;
}
