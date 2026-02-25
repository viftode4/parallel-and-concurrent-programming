#!/bin/bash
# run_benchmarks.sh — Run all versions with varying thread counts, verify correctness,
# and collect timing data (best of 3 runs for stability).
#
# Two baselines are used for fair comparison:
#   - sequential (int***) for the original parallel versions
#   - sequential_flat (contiguous) for the optimized parallel versions

set -e

BINDIR="bin"
THREADS="2 4 8 16"
RUNS=3
RESULTS_FILE="benchmark_results.csv"

EXPECTED_MIN="-1"
EXPECTED_MIN_POS="(499, 499, 499)"
EXPECTED_MAX="100000"
EXPECTED_MAX_POS="(250, 250, 250)"

echo "=== Building all versions ==="
make -j all
echo ""

check_correctness() {
    local label="$1" output="$2"
    local min_val min_pos max_val max_pos
    min_val=$(echo "$output" | grep "^Min" | sed 's/Min = \([^ ]*\).*/\1/')
    min_pos=$(echo "$output" | grep "^Min" | sed 's/.*at //')
    max_val=$(echo "$output" | grep "^Max" | sed 's/Max = \([^ ]*\).*/\1/')
    max_pos=$(echo "$output" | grep "^Max" | sed 's/.*at //')
    if [ "$min_val" != "$EXPECTED_MIN" ] || [ "$min_pos" != "$EXPECTED_MIN_POS" ] || \
       [ "$max_val" != "$EXPECTED_MAX" ] || [ "$max_pos" != "$EXPECTED_MAX_POS" ]; then
        echo "  FAIL [$label]: got Min=$min_val at $min_pos, Max=$max_val at $max_pos"
        return 1
    fi
    return 0
}

run_best_of() {
    local cmd="$1"
    BEST_TIME=""
    LAST_OUTPUT=""
    for r in $(seq 1 $RUNS); do
        LAST_OUTPUT=$(eval "$cmd")
        local t
        t=$(echo "$LAST_OUTPUT" | grep "Time:" | awk '{print $2}')
        if [ -z "$BEST_TIME" ] || [ "$(awk "BEGIN { print ($t < $BEST_TIME) }")" = "1" ]; then
            BEST_TIME="$t"
        fi
    done
}

echo "version,threads,time_seconds,baseline" > "$RESULTS_FILE"

# --- Baseline 1: sequential (pointer-of-pointer) ---
echo "=== Sequential baseline — int*** layout (best of $RUNS) ==="
run_best_of "\"$BINDIR/sequential\""
SEQ_PTR_TIME="$BEST_TIME"
echo "  Min/Max: $(echo "$LAST_OUTPUT" | head -2 | tr '\n' ', ')"
echo "  Time: $SEQ_PTR_TIME s"
check_correctness "sequential" "$LAST_OUTPUT"
echo "sequential,1,$SEQ_PTR_TIME,ptr" >> "$RESULTS_FILE"
echo ""

# --- Baseline 2: sequential_flat (contiguous) ---
echo "=== Sequential baseline — flat layout (best of $RUNS) ==="
run_best_of "\"$BINDIR/sequential_flat\""
SEQ_FLAT_TIME="$BEST_TIME"
echo "  Min/Max: $(echo "$LAST_OUTPUT" | head -2 | tr '\n' ', ')"
echo "  Time: $SEQ_FLAT_TIME s"
check_correctness "sequential_flat" "$LAST_OUTPUT"
echo "sequential_flat,1,$SEQ_FLAT_TIME,flat" >> "$RESULTS_FILE"
echo ""

# --- Original parallel versions (compared against sequential ptr) ---
ALL_PASS=true
for VERSION in version1_parallel_for version2_sections version3_combined; do
    echo "=== $VERSION (best of $RUNS) ==="
    for T in $THREADS; do
        run_best_of "OMP_NUM_THREADS=$T \"$BINDIR/$VERSION\""
        SP=$(awk "BEGIN { printf \"%.2f\", $SEQ_PTR_TIME / $BEST_TIME }")
        EF=$(awk "BEGIN { printf \"%.2f\", ($SEQ_PTR_TIME / $BEST_TIME) / $T }")
        if check_correctness "$VERSION T=$T" "$LAST_OUTPUT"; then
            printf "  T=%2d  Time=%-10s  Speedup=%-6s  Eff=%-6s  [PASS]\n" "$T" "$BEST_TIME" "$SP" "$EF"
        else
            printf "  T=%2d  Time=%-10s  Speedup=%-6s  Eff=%-6s  [FAIL]\n" "$T" "$BEST_TIME" "$SP" "$EF"
            ALL_PASS=false
        fi
        echo "$VERSION,$T,$BEST_TIME,ptr" >> "$RESULTS_FILE"
    done
    echo ""
done

# --- Optimized parallel versions (compared against sequential_flat) ---
for VERSION in version1_optimized version2_optimized version3_optimized; do
    echo "=== $VERSION (best of $RUNS) ==="
    for T in $THREADS; do
        run_best_of "OMP_NUM_THREADS=$T \"$BINDIR/$VERSION\""
        SP=$(awk "BEGIN { printf \"%.2f\", $SEQ_FLAT_TIME / $BEST_TIME }")
        EF=$(awk "BEGIN { printf \"%.2f\", ($SEQ_FLAT_TIME / $BEST_TIME) / $T }")
        if check_correctness "$VERSION T=$T" "$LAST_OUTPUT"; then
            printf "  T=%2d  Time=%-10s  Speedup=%-6s  Eff=%-6s  [PASS]\n" "$T" "$BEST_TIME" "$SP" "$EF"
        else
            printf "  T=%2d  Time=%-10s  Speedup=%-6s  Eff=%-6s  [FAIL]\n" "$T" "$BEST_TIME" "$SP" "$EF"
            ALL_PASS=false
        fi
        echo "$VERSION,$T,$BEST_TIME,flat" >> "$RESULTS_FILE"
    done
    echo ""
done

# --- Novel approaches (compared against sequential_flat) ---
for VERSION in novel_simd_avx2 novel_omp_simd novel_tiled novel_tasks novel_branchless novel_ultimate; do
    echo "=== $VERSION (best of $RUNS) ==="
    for T in $THREADS; do
        run_best_of "OMP_NUM_THREADS=$T \"$BINDIR/$VERSION\""
        SP=$(awk "BEGIN { printf \"%.2f\", $SEQ_FLAT_TIME / $BEST_TIME }")
        EF=$(awk "BEGIN { printf \"%.2f\", ($SEQ_FLAT_TIME / $BEST_TIME) / $T }")
        if check_correctness "$VERSION T=$T" "$LAST_OUTPUT"; then
            printf "  T=%2d  Time=%-10s  Speedup=%-6s  Eff=%-6s  [PASS]\n" "$T" "$BEST_TIME" "$SP" "$EF"
        else
            printf "  T=%2d  Time=%-10s  Speedup=%-6s  Eff=%-6s  [FAIL]\n" "$T" "$BEST_TIME" "$SP" "$EF"
            ALL_PASS=false
        fi
        echo "$VERSION,$T,$BEST_TIME,novel" >> "$RESULTS_FILE"
    done
    echo ""
done

# --- Summary tables ---
echo "============================================================"
echo "  ORIGINAL VERSIONS (baseline: sequential int***)"
echo "============================================================"
printf "%-28s %7s %10s %8s %10s\n" "Version" "Threads" "Time (s)" "Speedup" "Efficiency"
echo "------------------------------------------------------------------------"
printf "%-28s %7s %10s %8s %10s\n" "sequential" "1" "$SEQ_PTR_TIME" "1.00" "1.00"
while IFS=, read -r version threads time baseline; do
    [ "$version" = "version" ] || [ "$baseline" != "ptr" ] || [ "$version" = "sequential" ] && continue
    SP=$(awk "BEGIN { printf \"%.2f\", $SEQ_PTR_TIME / $time }")
    EF=$(awk "BEGIN { printf \"%.2f\", ($SEQ_PTR_TIME / $time) / $threads }")
    printf "%-28s %7s %10s %8s %10s\n" "$version" "$threads" "$time" "$SP" "$EF"
done < "$RESULTS_FILE"

echo ""
echo "============================================================"
echo "  OPTIMIZED VERSIONS (baseline: sequential_flat contiguous)"
echo "============================================================"
printf "%-28s %7s %10s %8s %10s\n" "Version" "Threads" "Time (s)" "Speedup" "Efficiency"
echo "------------------------------------------------------------------------"
printf "%-28s %7s %10s %8s %10s\n" "sequential_flat" "1" "$SEQ_FLAT_TIME" "1.00" "1.00"
while IFS=, read -r version threads time baseline; do
    [ "$version" = "version" ] || [ "$baseline" != "flat" ] || [ "$version" = "sequential_flat" ] && continue
    SP=$(awk "BEGIN { printf \"%.2f\", $SEQ_FLAT_TIME / $time }")
    EF=$(awk "BEGIN { printf \"%.2f\", ($SEQ_FLAT_TIME / $time) / $threads }")
    printf "%-28s %7s %10s %8s %10s\n" "$version" "$threads" "$time" "$SP" "$EF"
done < "$RESULTS_FILE"

echo ""
echo "============================================================"
echo "  NOVEL APPROACHES (baseline: sequential_flat contiguous)"
echo "============================================================"
printf "%-28s %7s %10s %8s %10s\n" "Version" "Threads" "Time (s)" "Speedup" "Efficiency"
echo "------------------------------------------------------------------------"
printf "%-28s %7s %10s %8s %10s\n" "sequential_flat" "1" "$SEQ_FLAT_TIME" "1.00" "1.00"
while IFS=, read -r version threads time baseline; do
    [ "$version" = "version" ] || [ "$baseline" != "novel" ] && continue
    SP=$(awk "BEGIN { printf \"%.2f\", $SEQ_FLAT_TIME / $time }")
    EF=$(awk "BEGIN { printf \"%.2f\", ($SEQ_FLAT_TIME / $time) / $threads }")
    printf "%-28s %7s %10s %8s %10s\n" "$version" "$threads" "$time" "$SP" "$EF"
done < "$RESULTS_FILE"

echo ""
if $ALL_PASS; then
    echo "All correctness checks PASSED."
else
    echo "WARNING: Some correctness checks FAILED!"
fi
echo "Raw results saved to $RESULTS_FILE"
