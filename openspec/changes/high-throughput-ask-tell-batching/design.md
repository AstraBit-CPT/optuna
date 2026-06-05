## Context

The existing ask/tell path is optimized around one trial at a time. `Study.ask()` invalidates cached trial history, claims or creates one trial, and `Trial._suggest()` calls the sampler and writes each sampled parameter. RDB trial creation locks the study row to allocate trial numbers, and TPE with `constant_liar=True` intentionally reads RUNNING trials so parallel suggestions avoid collapsing onto similar parameter regions.

Recent profiling of a low wall-clock-time workload showed the ask path split almost entirely between storage and sampler work:

- ask storage: 50.646%
- sampler suggestion: 49.2732%
- submit/transport: 0.0676%

That split means transport tuning or more workers cannot solve the core problem. The production update must amortize both trial allocation and sampler-history work, while preserving the invariants that make Optuna useful: unique trial identity, pending-trial awareness, idempotent completion, recovery after dead workers, and reproducible optimization history.

## Goals / Non-Goals

**Goals:**

- Add a high-throughput ask/tell plan for objectives where optimizer coordination can exceed objective runtime.
- Provide true batch reservation so multiple trial identities can be allocated in one coordination round.
- Provide pending-aware batch suggestion so samplers can emit multiple candidates from one history/model pass without dropping RUNNING trial semantics.
- Provide lease and idempotency semantics for workers that consume pre-reserved candidates.
- Provide an optional bounded candidate queue for millisecond objectives and saturated CPU/GPU workers.
- Provide cheap-generator modes or controls for cases where Random, QMC, or Sobol-style candidates outperform adaptive BO on best-value-per-second.
- Preserve existing `Study.ask()` and `Study.tell()` semantics by default.

**Non-Goals:**

- Do not make stale snapshots the default behavior for adaptive samplers.
- Do not treat leases as the throughput optimization by themselves.
- Do not promise sharded or island studies are equivalent to a single globally adaptive Bayesian study.
- Do not make GPU hardware scheduling part of sampler scoring; resource scheduling remains an execution concern.
- Do not require every storage backend to provide native bulk operations immediately. Backends can expose capability flags and conservative fallbacks.

## Decisions

- Introduce a core batch primitive instead of looping over public `ask()`.
  - Decision: Add a production-oriented API shape equivalent to `Study.ask_batch(count, ...)` plus `Study.tell_batch(...)`.
  - Rationale: A loop over `Study.ask()` repeats cache invalidation, storage allocation, sampler work, and parameter writes once per trial. It cannot remove the 50/49 wall.
  - Alternative considered: Add a helper that calls `ask()` repeatedly. Rejected because it is easier to adopt but does not change the bottleneck.

- Reserve trial identity in batches.
  - Decision: Add storage-level support equivalent to `create_new_trials(study_id, count, templates=None)` with unique trial IDs and trial numbers allocated under one storage coordination round.
  - Rationale: Trial identity must remain durable and unique before workers execute candidates.
  - Alternative considered: Generate transient candidate IDs outside storage. Rejected because completion, pruning, crash recovery, and audit history depend on durable trial identity.

- Add sampler batch hooks with pending-awareness.
  - Decision: Add a sampler extension point equivalent to `sample_batch(study, trials, search_space, pending)`; samplers without a native implementation may fall back to single-candidate behavior, but the high-throughput mode must report whether it is truly batched.
  - Rationale: The sampler side is roughly half the measured wall. TPE must be able to build or reuse its model once and treat in-batch candidates as pending/liar candidates before emitting later suggestions.
  - Alternative considered: Use stale completed-trial snapshots only. Rejected as a default because TPE `constant_liar` explicitly relies on RUNNING/pending trial visibility.

- Put bounded candidate queues above the batch primitive.
  - Decision: Add an opt-in coordinator that fills a small ready queue from `ask_batch`, leases candidates to workers, and applies results through `tell_batch` or idempotent single tells.
  - Rationale: For millisecond objectives, workers should not synchronously wait on storage and sampler work for every candidate. The coordinator moves that work off the worker critical path while bounding staleness.
  - Alternative considered: Unbounded prefetch. Rejected because it silently turns adaptive optimization into stale open-loop search.

- Treat leases as safety semantics, not the primary speedup.
  - Decision: Every queued candidate has a lease token, lease owner, lease deadline, and completion idempotency key.
  - Rationale: Leases solve abandoned workers, duplicate completion, restart recovery, and backpressure. They do not improve throughput unless paired with true batch reservation and batch sampling.
  - Alternative considered: Let workers hold raw trial IDs. Rejected because stale workers can double-complete or lose work without recoverable ownership metadata.

- Expose adaptiveness as an explicit tradeoff knob.
  - Decision: High-throughput mode records and enforces `max_inflight`, `candidate_queue_size`, `max_snapshot_age`, `batch_size`, and `generator_mode`.
  - Rationale: Other HPO systems make this tradeoff explicit: scheduler-first systems cap concurrency, batch systems trade adaptiveness for scale, and cheap candidate generators may beat BO for tiny objectives.
  - Alternative considered: Hide the tradeoff behind automatic heuristics. Rejected because users need to know when they are relaxing per-trial freshness.

- Keep resource scheduling separate from sampler semantics.
  - Decision: GPU or heterogeneous worker scheduling consumes ready candidates and reports utilization/backpressure separately from sampler decisions.
  - Rationale: Basic and advanced GPUs differ in memory and compute capacity. Saturation depends on execution batching and device assignment, not on changing sampler scoring.
  - Alternative considered: Include GPU capacity inside the sampler. Rejected because it couples optimization strategy to execution hardware and makes reproducibility harder.

## Risks / Trade-offs

- Batch suggestions can reduce sample efficiency -> Compare best-value-per-second and fixed-budget regret against synchronous TPE before promoting defaults.
- Stale candidate queues can duplicate or collapse suggestions -> Require pending/liar awareness, queue size limits, snapshot IDs, max age, and duplicate/near-duplicate metrics.
- Storage bulk reservation is backend-specific -> Add capability flags and use RDB as the first native target; label fallback implementations as correctness-only.
- `tell_batch` can obscure per-trial failures -> Record per-trial completion status, idempotency key, error state, and application order.
- Reproducibility can drift under asynchronous completion -> Persist batch ID, reservation order, sampler snapshot/hash, RNG state, lease token, and result application order.
- Cheap generators can waste samples in structured spaces -> Treat Random/QMC/Sobol as explicit generator modes and report quality tradeoffs, not as replacements for all BO use cases.
- Candidate queues add complexity -> Keep the minimal core batch API usable without the coordinator.

## Migration Plan

1. Implement benchmark-backed prototypes behind experimental APIs or feature flags.
2. Add RDB-native batch reservation first; keep other storage backends conservative until proven.
3. Add sampler batch hooks with default compatibility fallbacks and a native TPE implementation that preserves pending/liar semantics.
4. Add bounded candidate queue and lease recovery as an opt-in execution helper.
5. Document that existing `Study.ask()` and `Study.tell()` remain the default precise-per-trial path.
6. Promote defaults only after the research suite demonstrates throughput gains without unacceptable quality or recovery regressions.

Rollback is straightforward while the feature remains experimental: disable high-throughput mode and return to ordinary ask/tell. Storage changes must avoid irreversible schema requirements until the API is stable.

## Open Questions

- Should the public API be named `ask_batch`, `reserve_trials`, or a separate coordinator object?
- Should `tell_batch` accept mixed success/failure/pruned states in one call, or require homogeneous completion states?
- What is the minimum native storage interface needed to avoid overcommitting every backend?
- How should QMC/Sobol generator modes integrate with existing samplers and search-space inference?
- Which objective-duration threshold should documentation use to warn that adaptive BO overhead may dominate wall-clock results?
