## Why

Optuna's current ask/tell flow can serialize high-frequency trial allocation, parameter writes, sampler-history reads, and state updates in ways that become visible when 2, 4, 8, or more workers request short-running trials. We need a repeatable research suite before changing public APIs so the team can choose between batch reservation, leases, sampler-aware batching, storage/gRPC batching, and GPU-resource scheduling with evidence instead of intuition.

## What Changes

- Add a research experiment specification for measuring ask/tell throughput bottlenecks across worker counts, storage backends, sampler modes, and simulated heterogeneous GPU resources.
- Define benchmark scenarios for current ask/tell behavior, storage contention, batch reservation, lease recovery, sampler quality, and GPU saturation.
- Define output metrics and acceptance criteria that make lock contention, trial identity safety, optimization quality, and GPU utilization tradeoffs comparable.
- Produce an implementation task list for a CPU-safe harness with optional GPU/native extensions.
- No production API changes are introduced by this change.

## Capabilities

### New Capabilities

- `ask-tell-throughput-research-experiments`: Research harness and reporting requirements for Optuna ask/tell throughput, batch reservation, lease recovery, sampler-quality, and GPU-resource scheduling experiments.

### Modified Capabilities

- None.

## Impact

- Affects research and benchmark artifacts, not production Optuna runtime behavior.
- Experiment targets include `optuna/study/study.py`, `optuna/trial/_trial.py`, storage implementations, sampler behavior, and documentation caveats around parallel execution.
- No dependency changes are required for the default CPU-only experiment suite.
- Optional GPU/native experiment extensions must be guarded so they do not affect ordinary test or benchmark runs.
