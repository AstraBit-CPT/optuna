## Context

The current ask/tell path creates one trial at a time through `Study.ask`, resolves suggestions through `Trial._suggest`, and finalizes results through `Study.tell`. Local evidence shows several likely contention points: all-trial cache invalidation and history reads in `Study._get_trials`, waiting-trial selection in `_pop_waiting_trial_id`, RDB `for_update=True` sections during trial creation and state updates, broad JournalStorage `_thread_lock` sections, and TPE `constant_liar` handling of running trials.

The user-facing problem is not just a single lock. Short-running workers and GPU-native workloads can amplify every per-trial coordination round trip. Before changing Optuna APIs, storage contracts, or sampler behavior, the project needs experiments that compare throughput and optimization quality across current behavior and candidate designs.

## Goals / Non-Goals

**Goals:**

- Build a repeatable research suite that measures ask/tell overhead with 1, 2, 4, 8, and 16 workers.
- Compare baseline ask/tell behavior against simulated batch reservation, lease recovery, sampler-aware batching, storage/gRPC batching, and GPU-resource scheduling.
- Emit stable metrics that expose allocation latency, tell latency, trial throughput, lock/coordination wait, duplicate-trial safety, stale lease handling, sampler quality, and GPU utilization proxies.
- Keep the default suite CPU-only, deterministic, and safe for contributors without GPU hardware.
- Produce evidence that can drive a later production implementation decision.

**Non-Goals:**

- Do not introduce production APIs such as `Study.reserve`, `tell_batch`, storage batch hooks, or sampler `sample_batch` in this change.
- Do not require real GPU hardware for the default experiment suite.
- Do not treat SQLite as a target backend for high-throughput parallel optimization; use it only as a documented low-concurrency baseline.
- Do not change existing optimization semantics, storage schemas, or sampler contracts.

## Decisions

- Keep experiments separate from production runtime.
  - Decision: Place the harness under a benchmark or research-oriented path rather than in the core ask/tell implementation.
  - Rationale: The research suite must be able to model candidate designs without committing Optuna to API or storage contracts too early.
  - Alternative considered: Prototype `Study.reserve` directly in production code. Rejected because it would mix evidence gathering with API commitment.

- Use scenario-based CLI execution.
  - Decision: Provide named scenarios for `baseline-workers`, `storage-contention`, `batch-reservation`, `lease-recovery`, `sampler-quality`, and `gpu-saturation`.
  - Rationale: Named scenarios make experiments reproducible and allow focused runs during development.
  - Alternative considered: One monolithic benchmark. Rejected because it would make failures and regressions harder to isolate.

- Emit structured result files.
  - Decision: Write CSV or JSONL metrics for every scenario, plus a concise report summary.
  - Rationale: Structured data enables comparison across workers, backends, samplers, and candidate strategies.
  - Alternative considered: Human-readable logs only. Rejected because they are hard to aggregate and compare.

- Simulate candidate production designs first.
  - Decision: Model batch reservation, leases, and GPU scheduling in the harness before productionizing them.
  - Rationale: The key decision is whether these designs improve throughput without damaging trial identity or sampler quality.
  - Alternative considered: Benchmark only current behavior. Rejected because it would identify bottlenecks but not compare solutions.

- Keep GPU support layered.
  - Decision: The default GPU experiment is a resource-scheduling simulation with optional real GPU probes behind explicit flags.
  - Rationale: GPU saturation is a scheduling/resource-allocation problem adjacent to ask/tell, not sampler semantics.
  - Alternative considered: Make the sampler aware of GPU hardware. Rejected because it conflates optimization strategy with execution resources.

## Risks / Trade-offs

- Synthetic objectives may not represent real user workloads -> Include multiple objective durations and record overhead as a fraction of runtime.
- Simulated batch reservation may overstate production gains -> Report simulation assumptions and keep storage-backend baseline measurements separate.
- Measuring lock wait precisely may require intrusive instrumentation -> Start with latency decomposition and add optional deeper probes only when needed.
- GPU utilization proxies may miss hardware-specific bottlenecks -> Keep optional real-device runs separate from the CPU-only default suite.
- Parallel benchmark runs can be flaky -> Use deterministic seeds, bounded worker counts, explicit timeouts, and repeated runs with summary statistics.

## Migration Plan

This change creates research artifacts and a future implementation task list only. It does not require migration, rollback, schema changes, or user-facing behavior changes. If the later implementation adds production APIs or storage changes, that work should get a separate OpenSpec change based on the experiment results.

## Open Questions

- Which storage backend should be the primary high-concurrency target for the first production follow-up: RDB-only, gRPC-backed RDB, or a new coordinator layer?
- How much sampler staleness is acceptable for high-throughput GPU workloads before optimization quality becomes worse than the throughput gain?
- Which optional real GPU probes are worth adding once the CPU-only simulation produces a clear production direction?
