## 1. Harness Structure

- [ ] 1.1 Create `benchmarks/ask_tell_throughput/` with package files and a module entrypoint runnable with `python -m benchmarks.ask_tell_throughput`.
- [ ] 1.2 Add a scenario registry for `baseline-workers`, `storage-contention`, `batch-reservation`, `lease-recovery`, `sampler-quality`, and `gpu-saturation`.
- [ ] 1.3 Add shared configuration objects for worker counts, trial counts, objective durations, sampler options, storage options, output paths, and random seeds.
- [ ] 1.4 Add deterministic synthetic objective functions for fast, medium, and slow trial-duration profiles.
- [ ] 1.5 Add a bounded parallel worker runner that can execute ask/tell loops with 1, 2, 4, 8, and 16 workers.

## 2. Metrics and Reporting

- [ ] 2.1 Define a stable metrics schema for scenario name, worker count, storage mode, sampler mode, batch size, timing, throughput, errors, duplicate-trial count, stale lease count, and utilization proxies.
- [ ] 2.2 Implement CSV or JSONL metrics writing for every scenario run.
- [ ] 2.3 Implement summary aggregation with min, max, mean, median, and p95 timing fields for allocation and tell phases.
- [ ] 2.4 Implement a human-readable report that identifies the most promising candidate design and unresolved tradeoffs.
- [ ] 2.5 Add report sections for comparable patterns from Ray Tune, Vizier, Ax, Hyperopt SparkTrials, Nevergrad, BoTorch, KerasTuner, Syne Tune, and SMAC.

## 3. Baseline and Storage Experiments

- [ ] 3.1 Implement `baseline-workers` using current `Study.ask` and `Study.tell` behavior with deterministic objectives.
- [ ] 3.2 Record trial allocation latency, tell latency, total throughput, objective runtime, and overhead as a fraction of objective runtime.
- [ ] 3.3 Assert and report unique trial numbers and duplicate-trial count for baseline runs.
- [ ] 3.4 Implement `storage-contention` for configured Optuna storage modes available in the local environment.
- [ ] 3.5 Label SQLite results as a low-concurrency baseline in generated output and reports.
- [ ] 3.6 Keep storage-backend failures isolated so one unavailable backend does not prevent other scenarios from reporting results.

## 4. Candidate Design Simulations

- [ ] 4.1 Implement `batch-reservation` simulation with batch sizes 1, 2, 4, 8, and 16.
- [ ] 4.2 Record reservation latency, amortized per-trial allocation cost, throughput, duplicate-trial count, and staleness metrics for batch reservations.
- [ ] 4.3 Implement `lease-recovery` simulation with lease creation, heartbeat renewal, timeout, reclaim, and completion.
- [ ] 4.4 Verify lease simulation records one accepted completion and one rejected or ignored duplicate completion for competing completion attempts.
- [ ] 4.5 Implement `sampler-quality` comparisons for baseline ask/tell and pending-aware batch suggestion strategies.
- [ ] 4.6 Record best value over time, regret or rank-based quality metrics, throughput, and pending-trial count for sampler-quality runs.
- [ ] 4.7 Separate TPE `constant_liar` throughput impact from optimization-quality impact in reports.

## 5. GPU Resource Scheduling Simulation

- [ ] 5.1 Implement simulated GPU resource profiles for basic and advanced devices with configurable memory and compute capacities.
- [ ] 5.2 Implement a scheduler simulation that assigns trials to GPU profiles without changing sampler suggestion decisions.
- [ ] 5.3 Record assigned trials, delayed assignments, rejected assignments, memory pressure, compute pressure, and utilization proxy metrics per GPU profile.
- [ ] 5.4 Keep real GPU probing behind an explicit opt-in flag and skip it cleanly when GPU libraries or devices are unavailable.

## 6. Tests and Verification

- [ ] 6.1 Add unit tests for metrics schema stability and metrics file writing.
- [ ] 6.2 Add tests that default scenarios run on CPU-only machines without importing GPU-native packages.
- [ ] 6.3 Add tests for unique trial identity and duplicate-trial reporting in concurrent baseline runs.
- [ ] 6.4 Add tests for batch reservation simulation with no duplicate trial numbers.
- [ ] 6.5 Add tests for lease timeout, reclaim, and idempotent completion behavior.
- [ ] 6.6 Add tests for GPU scheduling simulation with mixed device profiles.
- [ ] 6.7 Run the relevant test subset and one short smoke run of the experiment CLI.
