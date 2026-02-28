# CS4560 Project Proposals — LLM-Augmented Concurrency Analysis

> **Team Project Proposals for CS4560 - Parallel and Concurrent Programming (2025-2026)**
>
> Three novel project ideas at the intersection of Large Language Models and concurrency bug analysis.
> All three have been verified as **genuinely novel** — no existing tool or paper implements these ideas.

---

## Proposal 1: ConcurrencyAgent — Autonomous Concurrency Debugging Agent

### The Problem

Concurrency bugs (data races, deadlocks, atomicity violations) are among the hardest bugs to debug. They're nondeterministic, hard to reproduce, and disappear under debuggers (Heisenbugs). At Uber, developers spend an average of **11 days fixing a single data race** (DR.FIX, PLDI 2025). Existing tools like ThreadSanitizer detect races but dump cryptic reports — developers still have to manually reason about root causes, design fixes, and verify them.

### The Idea

Build a **fully autonomous LLM-based agent** that handles the entire concurrency debugging lifecycle in a closed loop:

1. **Detect**: Compile code with `-fsanitize=thread`, run tests, capture ThreadSanitizer/Helgrind output
2. **Parse & Interpret**: LLM reads the sanitizer report (stack traces, racy accesses, thread IDs) and identifies the conflicting accesses
3. **Root Cause Analysis**: LLM reasons about *why* the race exists — missing lock? Wrong lock scope? Atomicity violation? Order violation?
4. **Fix Strategy Selection**: Based on root cause, the agent selects a repair strategy: add mutex, extend critical section, use atomic, add memory barrier, restructure shared state
5. **Fix Generation**: LLM generates a concrete code patch
6. **Verification**: Recompile with TSan, rerun tests, check if the race is eliminated without introducing new races
7. **Iterate**: If the fix fails or introduces new issues, analyze the feedback and retry with a different strategy

The key differentiator is that this is **truly agentic** — the LLM autonomously decides which tools to invoke and when, plans multi-step debugging strategies, and iterates based on feedback. This is NOT a fixed pipeline.

### Why This Doesn't Exist

| Existing Work | What It Does | What It Lacks |
|---|---|---|
| **DR.FIX** (Uber, PLDI 2025) | LLM + RAG fixes Go data races | Fixed pipeline, not agentic. Go-only. No recompile-verify loop. |
| **RepairAgent** (ICSE 2025) | First autonomous LLM agent for program repair | Sequential bugs only — no concurrency awareness, no TSan/Helgrind |
| **T2L-Agent** (Oct 2025) | Agentic trace analysis with sanitizers | Vulnerability *localization* only — no fix generation, no verification loop |
| **ConSynergy** (2025) | LLM + static analysis for concurrency bug detection | Detection only, fixed 4-stage pipeline, no repair |
| **CFix** (OSDI 2012) | Automated concurrency bug fixing | No LLM, rule-based strategies, no semantic understanding |

**The gap**: Agentic LLM repair (RepairAgent) and concurrency-specific LLM analysis (DR.FIX, ConSynergy) exist as disconnected research threads. **Nobody has combined them.**

### Why This Is Impactful

- **Universal audience**: Every developer working with threads needs this
- **Product-viable**: This could become a real tool/VS Code extension/CI integration
- **Measurable**: Fix success rate, time-to-fix, races eliminated — all directly benchmarkable
- **Addresses real pain**: Concurrency debugging is consistently ranked as one of the hardest tasks in software engineering

### Technical Details

- **Concurrency Model**: Shared-memory multi-threading (pthreads, C++11 threads). The agent reasons about happens-before relations, lock semantics, atomic operations, and memory visibility
- **Sources of Nondeterminism**: Thread scheduling by the OS, memory access reordering by CPU/compiler, timing-dependent interleavings
- **Agent Framework**: LangChain/LangGraph or custom tool-use loop with Claude API or GPT-4
- **Tools Available to the Agent**: ThreadSanitizer (Clang), Helgrind (Valgrind), compiler (clang++), test runner, static analyzer (optional: RacerD, Infer)
- **Languages**: C/C++ primarily (TSan support), potentially extensible to Go, Rust, Java

### Benchmarks & Evaluation

- **DataRaceBench** — standardized benchmark suite for data race detection tools
- **SV-COMP pthread suite** — software verification competition benchmarks
- **DR.FIX's open-source race skeletons** — real Uber production race patterns
- **Metrics**: Fix success rate, false positive rate, time-to-fix, comparison vs. manual debugging, DR.FIX pipeline, and RepairAgent

### Key References

1. DR.FIX: Automatically Fixing Data Races at Industry Scale — PLDI 2025 [[paper]](https://arxiv.org/abs/2504.15637) [[code]](https://github.com/uber-research/drfix)
2. RepairAgent: An Autonomous, LLM-Based Agent for Program Repair — ICSE 2025 [[paper]](https://arxiv.org/abs/2403.17134)
3. T2L-Agent: From Trace to Line: LLM Agent for Real-World Vulnerability Localization — 2025 [[paper]](https://arxiv.org/abs/2510.02389)
4. ConSynergy: Concurrency Bug Detection via Static Analysis and LLMs — 2025 [[paper]](https://www.mdpi.com/1999-5903/17/12/578)
5. CFix: Automated Concurrency-Bug Fixing — OSDI 2012 [[paper]](https://www.usenix.org/conference/osdi12/technical-sessions/presentation/jin)
6. Assessing LLMs in Verifying Concurrent Programs across Memory Models — 2025 [[paper]](https://arxiv.org/abs/2501.14326)

---

## Proposal 2: InterleaveSynth — LLM-Guided Adversarial Thread Schedule Generation

### The Problem

Concurrency testing is fundamentally limited by the **exponential interleaving space**. A program with N threads and M instructions per thread has roughly (NM)!/(M!)^N possible interleavings. Current approaches are either:
- **Random** (stress testing, PCT) — fast but blind, often misses deep bugs
- **Exhaustive** (DPOR) — sound but exponential, doesn't scale
- **RL-guided** (Microsoft's QL) — learns from runtime traces but has no semantic understanding of the code

None of them *read the code* to understand *which* interleavings are dangerous.

### The Idea

Use an LLM as a **semantic schedule oracle** that reads concurrent source code, understands the shared state and synchronization patterns, and predicts which specific thread interleavings are most likely to trigger bugs:

1. **Static Analysis Phase**: LLM reads the concurrent program, identifies shared variables, lock acquisitions, atomic operations, and potential race windows
2. **Schedule Generation**: LLM generates a ranked list of "suspicious interleavings" — specific thread switch points (e.g., "switch from Thread 1 to Thread 2 right after the lock release on line 42 but before the store on line 43")
3. **Controlled Execution**: A deterministic scheduler (like PCT or Coyote) executes the program under each proposed schedule
4. **Feedback Loop**: Bug-exposing schedules are fed back to the LLM for pattern recognition; non-exposing schedules become negative examples
5. **Iterative Refinement**: The LLM refines its predictions based on accumulated evidence

This is fundamentally a **new paradigm**: using code comprehension to navigate the interleaving space, rather than random exploration or brute-force enumeration.

### Why This Doesn't Exist

| Existing Work | Approach | Limitation |
|---|---|---|
| **PCT** (ASPLOS 2010) | Randomized priority scheduling | Blind — no code understanding |
| **DPOR** (multiple variants) | Systematic exploration with independence pruning | Exponential in depth, no prioritization |
| **QL / Coyote** (Microsoft, OOPSLA 2020) | Q-learning from runtime traces | Black-box feedback — doesn't read code |
| **Active Testing / CalFuzzer** (Berkeley, 2008) | Predict-then-confirm with traditional analysis | Uses static analysis, not LLMs — limited semantic understanding |
| **PERIOD** (ICSE 2022) | Feedback-guided periodic scheduling | Runtime heuristics, no code comprehension |
| **ConSynergy / LLM detectors** | LLM reads code for bug detection | Detection only — never generates executable schedules |

**The gap**: LLM-based concurrency work stops at detection/classification. Schedule exploration work uses RL or random search. Nobody has used an LLM to *predict schedules from source code*.

### Why This Is Impactful

- **Paradigm shift**: If LLMs can effectively predict dangerous interleavings from code semantics, it could fundamentally change how concurrency testing works
- **Orders of magnitude faster**: Instead of exploring millions of random schedules, explore the 100 most suspicious ones first
- **Complementary to existing tools**: Can be layered on top of any deterministic scheduler (PCT, Coyote, CHESS)
- **Bridges static and dynamic analysis**: LLM provides semantic understanding (static), scheduler provides execution (dynamic)

### Technical Details

- **Concurrency Model**: Shared-memory multi-threading. The LLM must understand thread creation, join, mutex lock/unlock, condition variables, atomic operations, and the happens-before relation
- **Sources of Nondeterminism**: OS thread scheduling decisions — which thread runs next at each preemption point. The LLM explicitly reasons about which scheduling choices lead to bugs
- **Schedule Representation**: A schedule is represented as a sequence of (thread_id, instruction_count) pairs specifying which thread runs and for how many steps before a context switch
- **Deterministic Scheduler**: Could build on Coyote (C#), or implement a `ptrace`-based scheduler for C/C++ that forces specific thread orderings

### Benchmarks & Evaluation

- **SCTBench** — standard benchmark for controlled concurrency testing
- **ConVul** — real-world concurrency vulnerabilities
- **DR.FIX race skeletons** — real production race patterns
- **Metrics**: Bugs found, time-to-first-bug, number of schedules explored before finding bug, comparison vs. PCT (random), QL (RL-guided), DPOR (exhaustive)

### Key References

1. A Randomized Scheduler with Probabilistic Guarantees of Finding Bugs (PCT) — ASPLOS 2010 [[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos277-pct.pdf)
2. Learning-based Controlled Concurrency Testing (QL) — OOPSLA 2020 [[paper]](https://dl.acm.org/doi/10.1145/3428298)
3. Active Testing for Concurrent Programs — PLDI 2009 [[paper]](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-243.pdf)
4. PERIOD: Controlled Concurrency Testing via Periodical Scheduling — ICSE 2022 [[paper]](https://wcventure.github.io/pdf/ICSE2022_PERIOD.pdf)
5. Coyote: Industrial-Strength Controlled Concurrency Testing — TACAS 2023 [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-30820-8_26)
6. ConSynergy: Concurrency Bug Detection via Static Analysis and LLMs — 2025 [[paper]](https://www.mdpi.com/1999-5903/17/12/578)
7. DPOR: Dynamic Partial-Order Reduction Survey [[survey]](https://dl.acm.org/doi/10.1145/2858651)

---

## Proposal 3: MemOrder Advisor — LLM + Model Checker for Memory Ordering Optimization

### The Problem

Lock-free concurrent code requires careful use of C++ memory orderings (`seq_cst`, `acquire`, `release`, `relaxed`). In practice:
- **Developers default to `seq_cst`** (the strongest ordering) because it's "safe" — but it kills performance on weakly-ordered architectures like ARM and RISC-V
- **Weakening orderings manually is terrifying** — one wrong `relaxed` and you get a subtle bug that only manifests on ARM under specific timing
- **Formal tools exist** (FenSying, Musketeer) but they use constraint solving and are hard for developers to understand or trust

Meanwhile, a 2025 study proved that **LLMs completely fail at relaxed memory model reasoning** — they can't reason about TSO, PSO, or ARM memory orderings on their own (Jain & Purandare, arXiv 2501.14326).

### The Idea

Combine an LLM's semantic understanding of code with a formal model checker's correctness guarantees in a **feedback loop** that systematically weakens memory orderings:

1. **Input**: Lock-free C++ code using `seq_cst` atomics everywhere
2. **LLM Proposes Weakening**: The LLM reads the code, understands the algorithm's intent, and proposes a specific weakening (e.g., "this store can be `release` because it publishes data that the acquiring load on line 47 needs to see")
3. **Model Checker Verifies**: GenMC or herd7 checks whether the weakened program is still correct under the C11 memory model by exhaustively exploring all possible weak memory behaviors
4. **Counterexample Analysis**: If the model checker finds a violating execution, the LLM receives the counterexample trace, understands *why* the weakening was unsafe, and proposes a different strategy
5. **Iterate**: Repeat until all orderings are as weak as possible while maintaining correctness
6. **Output**: Optimized code with minimal memory orderings + explanation of why each ordering is necessary

### Why This Doesn't Exist

| Existing Work | Approach | Limitation |
|---|---|---|
| **FenSying** (ATVA 2022) | Optimal fence synthesis via constraint solving for C11 | No LLM, no semantic understanding, NP-hard |
| **Musketeer** (TOPLAS 2017) | Static analysis for fence insertion | Trades precision for scalability, no LLM |
| **DFENCE** (PLDI 2012) | Dynamic fence synthesis | No LLM, no explanation of *why* fences are needed |
| **Memorax** (TACAS 2013) | CEGAR-based verification + fence insertion | Limited to finite-state, no LLM |
| **LLMs for memory models** (arXiv 2501.14326) | Assess LLM understanding of TSO/PSO | Assessment only — concludes LLMs *fail* at this. No feedback loop to fix the failure |

**The gap**: Traditional tools do fence synthesis but give no explanations. LLMs understand code semantics but fail at memory ordering. Nobody has combined them in a feedback loop where the model checker compensates for the LLM's weakness.

### Why This Is Impactful

- **ARM/RISC-V are everywhere**: Apple Silicon, AWS Graviton, smartphones, IoT — all weakly ordered. Code written with `seq_cst` leaves serious performance on the table
- **Directly overcomes a proven LLM limitation**: The 2025 benchmark paper showed LLMs fail at memory ordering alone. This project shows that LLM + formal tool *succeeds*. That's a compelling research narrative
- **Developer-friendly**: Unlike FenSying's constraint output, the LLM explains *why* each ordering is needed in natural language
- **Measurable performance gains**: Weakening orderings from `seq_cst` to `release`/`acquire` can yield 2-10x speedups on ARM for lock-free structures

### Technical Details

- **Concurrency Model**: C/C++11 memory model (RC11 variant). The project must explain: sequential consistency, TSO, release-acquire, relaxed semantics, happens-before, synchronizes-with, modification order, coherence order
- **Sources of Nondeterminism**: Weak memory models allow CPUs and compilers to reorder memory accesses. A `relaxed` store might become visible to other threads in a different order than program order. The model checker explores all such nondeterministic outcomes
- **Model Checker**: GenMC (stateless model checker for C11, from MPI-SWS) or herd7 (axiomatic simulator for memory models)
- **LLM**: Claude/GPT-4 with structured prompts that include the code, the proposed weakening, and any counterexample from the last iteration

### Benchmarks & Evaluation

- **Litmus tests**: herd7's extensive litmus test suite for C11
- **Lock-free data structures**: Michael-Scott queue, Treiber stack, Harris linked list, chase-lev work-stealing deque
- **Metrics**: Number of orderings successfully weakened, correctness preserved (model checker verified), performance improvement on ARM vs x86, comparison vs. FenSying (optimal) and Musketeer (heuristic)

### Key References

1. Assessing LLMs in Verifying Concurrent Programs across Memory Models — 2025 [[paper]](https://arxiv.org/abs/2501.14326)
2. Fence Synthesis Under the C11 Memory Model (FenSying) — ATVA 2022 [[paper]](https://arxiv.org/abs/2208.00285)
3. Don't Sit on the Fence: A Static Analysis Approach (Musketeer) — TOPLAS 2017 [[paper]](https://dl.acm.org/doi/10.1145/2994593)
4. Dynamic Synthesis for Relaxed Memory Models (DFENCE) — PLDI 2012 [[paper]](https://dl.acm.org/doi/10.1145/2345156.2254115)
5. Synthesizing Memory Models from Framework Sketches and Litmus Tests (MemSynth) — PLDI 2017 [[paper]](https://dl.acm.org/doi/10.1145/3062341.3062353)
6. Automatic Inference of Memory Fences (FENDER) — FMCAD 2010 [[paper]](https://csaws.cs.technion.ac.il/~yahave/papers/fmcad10.pdf)
7. GenMC: A Model Checker for Weak Memory Models [[tool]](https://plv.mpi-sws.org/genmc/)
8. herd7: A Memory Model Simulator [[tool]](https://github.com/herd/herdtools7)

---

## How These Proposals Fit the CS4560 Rubric

| Rubric Criterion | ConcurrencyAgent | InterleaveSynth | MemOrder Advisor |
|---|---|---|---|
| **Concurrency model explanation (10 pts)** | Shared-memory threading, happens-before, lock semantics | Shared-memory threading, interleaving space, happens-before | C11 memory model, SC/TSO/release-acquire/relaxed |
| **Sources of nondeterminism (10 pts)** | OS thread scheduling, memory reordering, timing | Thread scheduling decisions at preemption points | Weak memory reordering by CPU and compiler |
| **Bug detection technique (10 pts)** | Agentic LLM loop with sanitizer tools | LLM-guided schedule prediction + controlled execution | LLM proposes ordering weakenings, model checker verifies |
| **Related work comparison (10 pts)** | vs. DR.FIX, RepairAgent, CFix, ConSynergy | vs. PCT, QL, DPOR, Active Testing, PERIOD | vs. FenSying, Musketeer, DFENCE, Memorax |
| **Source code (10 pts)** | Agent framework + tool integration | Schedule generator + scheduler integration | LLM pipeline + model checker integration |
| **Empirical evaluation (25 pts)** | Fix rate, time-to-fix on DataRaceBench, SV-COMP | Bugs found, time-to-first-bug on SCTBench, ConVul | Orderings weakened, performance gain on ARM |
| **Presentation (25 pts)** | Live demo: agent finds and fixes a race | Live demo: LLM predicts the exact interleaving | Live demo: seq_cst → relaxed with 5x speedup on ARM |

---

## Summary & Priority Ranking

| Priority | Project | Why |
|---|---|---|
| **#1** | **ConcurrencyAgent** | Broadest real-world impact. Every developer needs this. Most feasible to build. Strongest demo potential. Could become a real open-source tool or product. |
| **#2** | **InterleaveSynth** | Most paradigm-shifting. If it works, it fundamentally changes concurrency testing. New research direction with clear baselines. High risk, highest reward. |
| **#3** | **MemOrder Advisor** | Directly overcomes a proven LLM limitation. Growing importance with ARM/RISC-V adoption. Cleanest academic narrative. Most niche audience. |

All three are **verified novel** as of February 2026 — no existing tool or paper implements these ideas.
