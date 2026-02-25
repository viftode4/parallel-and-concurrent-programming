/*
 * Novel Approach: Task-Based Recursive Divide & Conquer
 *
 * Instead of parallel for with static scheduling, recursively splits the
 * array in half and creates OpenMP tasks for each half. Results bubble up
 * through the recursion tree — a natural parallel reduction.
 *
 * Uses #pragma omp task with:
 *   - shared() for result variables
 *   - firstprivate() for array pointer and bounds
 *   - final() to stop spawning tasks below a threshold (avoids explosion)
 *   - taskwait to synchronize before merging results
 *
 * The work-stealing scheduler automatically load-balances across threads.
 * A #pragma omp single inside parallel launches the root task; all threads
 * participate in the task pool via work-stealing.
 *
 * Expected: roughly on par with parallel for. Main value is demonstrating
 * the task paradigm and composability.
 *
 * Compile with: gcc -O2 -fopenmp
 */
#include "common.h"

/* Threshold: below this, scan sequentially (fits in L2 ~256 KB = 64K ints) */
#define TASK_THRESHOLD 65536

typedef struct { int val; long idx; } FlatResult;

static FlatResult task_find_min(const int *a, long lo, long hi)
{
    if (hi - lo <= TASK_THRESHOLD) {
        FlatResult r = { INT_MAX, lo };
        for (long i = lo; i < hi; i++) {
            if (a[i] < r.val) { r.val = a[i]; r.idx = i; }
        }
        return r;
    }

    long mid = lo + (hi - lo) / 2;
    FlatResult left, right;

    #pragma omp task shared(left) firstprivate(a, lo, mid) \
                     final((mid - lo) <= TASK_THRESHOLD)
    left = task_find_min(a, lo, mid);

    #pragma omp task shared(right) firstprivate(a, mid, hi) \
                     final((hi - mid) <= TASK_THRESHOLD)
    right = task_find_min(a, mid, hi);

    #pragma omp taskwait

    return (left.val <= right.val) ? left : right;
}

static FlatResult task_find_max(const int *a, long lo, long hi)
{
    if (hi - lo <= TASK_THRESHOLD) {
        FlatResult r = { INT_MIN, lo };
        for (long i = lo; i < hi; i++) {
            if (a[i] > r.val) { r.val = a[i]; r.idx = i; }
        }
        return r;
    }

    long mid = lo + (hi - lo) / 2;
    FlatResult left, right;

    #pragma omp task shared(left) firstprivate(a, lo, mid) \
                     final((mid - lo) <= TASK_THRESHOLD)
    left = task_find_max(a, lo, mid);

    #pragma omp task shared(right) firstprivate(a, mid, hi) \
                     final((hi - mid) <= TASK_THRESHOLD)
    right = task_find_max(a, mid, hi);

    #pragma omp taskwait

    return (left.val >= right.val) ? left : right;
}

int main(void)
{
    int *a;
    int M, N, P;
    read_input_flat(&a, &M, &N, &P);
    long total = (long)M * N * P;

    FlatResult gmin, gmax;

    double t_start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            /* Launch min and max searches as two independent top-level tasks */
            #pragma omp task shared(gmin)
            gmin = task_find_min(a, 0, total);

            #pragma omp task shared(gmax)
            gmax = task_find_max(a, 0, total);
        }
        /* Implicit barrier at end of single */
    }

    double t_end = omp_get_wtime();

    /* Convert flat index to 3D */
    int mi = (int)(gmin.idx / ((long)N * P));
    int mj = (int)((gmin.idx % ((long)N * P)) / P);
    int mk = (int)(gmin.idx % P);

    int xi = (int)(gmax.idx / ((long)N * P));
    int xj = (int)((gmax.idx % ((long)N * P)) / P);
    int xk = (int)(gmax.idx % P);

    printf("Min = %d at (%d, %d, %d)\n", gmin.val, mi, mj, mk);
    printf("Max = %d at (%d, %d, %d)\n", gmax.val, xi, xj, xk);
    printf("Time: %.6f seconds\n", t_end - t_start);

    free(a);
    return 0;
}
