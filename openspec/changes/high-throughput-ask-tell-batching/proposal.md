## Why

Low wall-clock-time objectives can make Optuna's per-trial ask/suggest/tell coordination dominate the workload. Recent profiling showed the ask path split across storage and sampler work, with submit/transport effectively negligible, so higher worker counts alone will not remove the bottleneck.

Optuna needs a planned high-throughput ask/tell mode that preserves trial identity, pending-trial awareness, recovery, and reproducibility while giving users an explicit throughput-versus-adaptiveness tradeoff for millisecond objectives and saturated CPU/GPU workers.

## What Changes

- Add a production API plan for reserving and suggesting multiple trials per coordination round instead of requiring every worker to synchronously call `Study.ask`.
- Define pending-aware batch suggestion semantics that preserve or intentionally replace TPE `constant_liar` behavior for RUNNING trials.
- Define worker-facing candidate leases so pre-reserved trials can be consumed, completed, expired, and recovered safely.
- Define bounded candidate queues as an opt-in hot-path mode for very short objectives, with max age, max depth, duplicate protection, and reproducibility metadata.
- Define cheap generator controls for cases where Random, QMC, or Sobol-style candidates produce better best-value-per-second than adaptive Bayesian suggestions.
- Keep GPU/device scheduling separate from sampler semantics while allowing candidate production to keep heterogeneous workers saturated.

## Capabilities

### New Capabilities

- `high-throughput-ask-tell-batching`: Batch reservation, pending-aware candidate generation, worker leases, bounded candidate queues, and reporting/metadata requirements for high-throughput ask/tell usage.

### Modified Capabilities

None.

## Impact

- Affected APIs: `Study.ask`, `Study.tell`, future batch reservation/candidate APIs, and documentation for ask/tell parallel usage.
- Affected internals: `Study` trial allocation, `Trial._suggest`, sampler interfaces, storage trial creation/state updates, heartbeat or lease recovery behavior, and benchmark/reporting guidance.
- Affected storage backends: RDB-backed storage is the primary target; SQLite remains a low-concurrency baseline, and gRPC-backed storage is a transport/scaling variant rather than the core fix.
- Affected users: users optimizing millisecond-to-subsecond objectives, distributed workers, and CPU/GPU native workloads where optimizer coordination can exceed objective runtime.
