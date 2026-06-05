## ADDED Requirements

### Requirement: CPU-only default experiment suite
The experiment suite SHALL run its default scenarios without requiring GPU hardware or optional GPU-native dependencies.

#### Scenario: Run default suite on CPU-only machine
- **WHEN** a contributor runs the default experiment suite on a machine with no GPU
- **THEN** the suite completes the default scenarios and writes result artifacts without requiring GPU-specific packages

#### Scenario: Optional GPU probes are disabled by default
- **WHEN** a contributor runs the suite without an explicit GPU flag
- **THEN** no real GPU device probing or GPU-native execution is attempted

### Requirement: Baseline worker scaling experiment
The experiment suite SHALL measure current ask/tell behavior across multiple worker counts using deterministic synthetic objectives.

#### Scenario: Run baseline worker counts
- **WHEN** the baseline worker scenario is run
- **THEN** results are recorded for 1, 2, 4, 8, and 16 workers

#### Scenario: Preserve trial identity during baseline
- **WHEN** baseline workers concurrently request and complete trials
- **THEN** every completed trial has a unique trial number and no duplicate trial identity is reported

### Requirement: Storage contention experiment
The experiment suite SHALL compare ask/tell coordination overhead across storage configurations relevant to current Optuna behavior.

#### Scenario: Compare supported storage paths
- **WHEN** the storage contention scenario is run
- **THEN** the suite records allocation latency, tell latency, throughput, and failure counts for each configured storage path

#### Scenario: Include SQLite only as low-concurrency baseline
- **WHEN** SQLite is included in storage contention results
- **THEN** the report labels SQLite as a documented low-concurrency baseline rather than a target high-throughput backend

### Requirement: Batch reservation simulation
The experiment suite SHALL simulate reserving multiple trials per coordination round trip and compare it with single-trial ask behavior.

#### Scenario: Reserve trial batches
- **WHEN** the batch reservation scenario runs with a configured batch size
- **THEN** the suite records reservation latency, per-trial amortized allocation cost, throughput, and duplicate-trial count

#### Scenario: Compare batch size tradeoffs
- **WHEN** batch sizes of 1, 2, 4, 8, and 16 are tested
- **THEN** the report compares throughput gains against staleness and sampler-quality metrics

### Requirement: Lease recovery simulation
The experiment suite SHALL simulate trial reservation leases, heartbeat renewal, timeout, reclaim, and idempotent completion.

#### Scenario: Reclaim stale reservation
- **WHEN** a simulated worker stops renewing a trial lease past the configured timeout
- **THEN** the trial is marked reclaimable and the event is recorded in the metrics

#### Scenario: Reject duplicate completion
- **WHEN** two simulated workers attempt to complete the same leased trial
- **THEN** the suite records one accepted completion and one rejected or ignored duplicate completion

### Requirement: Sampler quality experiment
The experiment suite SHALL measure both throughput and optimization quality for baseline and pending-aware batch suggestion strategies.

#### Scenario: Compare sampler strategies
- **WHEN** the sampler quality scenario runs
- **THEN** the suite records best value over time, regret or rank-based quality metrics, throughput, and pending-trial count for each strategy

#### Scenario: Report constant liar impact
- **WHEN** TPE constant-liar behavior is part of the scenario configuration
- **THEN** the report separates throughput impact from optimization-quality impact

### Requirement: GPU resource scheduling simulation
The experiment suite SHALL model heterogeneous GPU workers with different memory and compute capacities without requiring real GPU hardware.

#### Scenario: Schedule mixed GPU profiles
- **WHEN** the GPU saturation scenario runs with basic and advanced simulated GPU profiles
- **THEN** the suite records assigned trials, rejected or delayed assignments, memory pressure, compute pressure, and utilization proxy metrics per profile

#### Scenario: Keep scheduler separate from sampler
- **WHEN** GPU scheduling simulation assigns work
- **THEN** the simulated resource decision is recorded separately from sampler suggestion decisions

### Requirement: Structured metrics and summary report
The experiment suite SHALL emit stable structured metrics and a human-readable report for every scenario run.

#### Scenario: Write structured metrics
- **WHEN** any experiment scenario completes
- **THEN** the suite writes CSV or JSONL metrics with stable field names for scenario name, worker count, storage mode, sampler mode, timing, throughput, and error counts

#### Scenario: Write decision summary
- **WHEN** the full suite completes
- **THEN** the suite writes a report summarizing which candidate design appears most promising and what risks remain before production implementation

### Requirement: External pattern comparison
The experiment report SHALL compare Optuna candidate designs with relevant patterns from similar optimization and tuning systems.

#### Scenario: Include comparable libraries
- **WHEN** the report is generated
- **THEN** it references the relevant design patterns from Ray Tune, Vizier, Ax, Hyperopt SparkTrials, Nevergrad, BoTorch, KerasTuner, Syne Tune, and SMAC

#### Scenario: Separate evidence from recommendation
- **WHEN** the report recommends a production follow-up
- **THEN** the recommendation identifies the supporting experiment results and the tradeoffs that remain unresolved
