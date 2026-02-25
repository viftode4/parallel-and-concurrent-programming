CC = gcc
CFLAGS = -O2 -fopenmp -Wall
SRCDIR = src
BINDIR = bin

TARGETS = $(BINDIR)/sequential \
          $(BINDIR)/sequential_flat \
          $(BINDIR)/version1_parallel_for \
          $(BINDIR)/version2_sections \
          $(BINDIR)/version3_combined \
          $(BINDIR)/version1_optimized \
          $(BINDIR)/version2_optimized \
          $(BINDIR)/version3_optimized \
          $(BINDIR)/novel_simd_avx2 \
          $(BINDIR)/novel_omp_simd \
          $(BINDIR)/novel_tiled \
          $(BINDIR)/novel_tasks \
          $(BINDIR)/novel_branchless \
          $(BINDIR)/novel_ultimate

.PHONY: all clean

all: $(BINDIR) $(TARGETS)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/sequential: $(SRCDIR)/sequential.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/sequential_flat: $(SRCDIR)/sequential_flat.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/version1_parallel_for: $(SRCDIR)/version1_parallel_for.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/version2_sections: $(SRCDIR)/version2_sections.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/version3_combined: $(SRCDIR)/version3_combined.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/version1_optimized: $(SRCDIR)/version1_optimized.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/version2_optimized: $(SRCDIR)/version2_optimized.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/version3_optimized: $(SRCDIR)/version3_optimized.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

# --- Novel approaches ---

$(BINDIR)/novel_simd_avx2: $(SRCDIR)/novel_simd_avx2.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -mavx2 -o $@ $<

$(BINDIR)/novel_omp_simd: $(SRCDIR)/novel_omp_simd.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -mavx2 -o $@ $<

$(BINDIR)/novel_tiled: $(SRCDIR)/novel_tiled.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/novel_tasks: $(SRCDIR)/novel_tasks.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/novel_branchless: $(SRCDIR)/novel_branchless.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -o $@ $<

$(BINDIR)/novel_ultimate: $(SRCDIR)/novel_ultimate.c $(SRCDIR)/common.h
	$(CC) $(CFLAGS) -mavx2 -o $@ $<

clean:
	rm -rf $(BINDIR)
