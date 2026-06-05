## ADDED Requirements

### Requirement: Batch trial reservation
Optuna SHALL provide a high-throughput path that can reserve multiple trial identities in one coordination round while preserving unique durable trial IDs and trial numbers.

#### Scenario: Reserve a batch of trials
- **WHEN** a caller reserves a batch of N trials for a study
- **THEN** Optuna returns N distinct trial handles with unique trial IDs and trial numbers

#### Scenario: Persist trial identity before execution
- **WHEN** a worker receives a reserved trial handle
- **THEN** the trial identity is durable enough for completion, failure, pruning, or recovery after a coordinator restart

#### Scenario: Report non-native fallback
- **WHEN** a storage backend cannot reserve trials in one native operation
- **THEN** Optuna reports that the backend is using a correctness fallback rather than claiming high-throughput reservation support

### Requirement: Pending-aware batch suggestion
Optuna SHALL support batch candidate generation that preserves pending-trial awareness for samplers that rely on RUNNING trial visibility.

#### Scenario: Generate pending-aware batch suggestions
- **WHEN** a sampler emits multiple suggestions for one reserved batch
- **THEN** each later in-batch suggestion accounts for earlier in-batch suggestions as pending candidates

#### Scenario: Preserve constant-liar semantics
- **WHEN** TPE constant-liar behavior is enabled during batch suggestion
- **THEN** RUNNING and newly batched pending trials are considered according to the constant-liar policy before suggestions are returned to workers

#### Scenario: Distinguish true batching from repeated ask
- **WHEN** a high-throughput batch suggestion run completes
- **THEN** Optuna records whether the sampler used a native batch path or a repeated single-suggestion fallback

### Requirement: Batch completion
Optuna SHALL support applying multiple trial completions through a batch-aware path while preserving per-trial outcomes and compatibility with existing tell semantics.

#### Scenario: Complete a batch with values
- **WHEN** a caller submits completion results for multiple reserved trials
- **THEN** Optuna records each trial's values, state, completion time, and error status independently

#### Scenario: Preserve single-trial tell compatibility
- **WHEN** a user continues to call existing single-trial `tell`
- **THEN** Optuna preserves existing `tell` behavior without requiring batch mode

#### Scenario: Reject duplicate completion
- **WHEN** the same reserved trial is completed twice with the same or competing completion token
- **THEN** Optuna accepts at most one completion and records or reports the duplicate attempt

### Requirement: Worker lease recovery
Optuna SHALL provide lease semantics for pre-reserved or queued candidates so abandoned work can be reclaimed without corrupting trial history.

#### Scenario: Lease a candidate to a worker
- **WHEN** a worker consumes a pre-reserved candidate
- **THEN** Optuna associates the candidate with a lease owner, lease token, and lease deadline

#### Scenario: Reclaim expired lease
- **WHEN** a worker stops renewing a candidate lease past the configured deadline
- **THEN** Optuna marks the candidate reclaimable or transitions it according to the configured recovery policy

#### Scenario: Complete with valid lease token
- **WHEN** a worker completes a leased candidate with the current valid lease token
- **THEN** Optuna accepts the completion and invalidates future completions for that lease token

### Requirement: Bounded candidate queue
Optuna SHALL provide an opt-in bounded candidate queue mode that allows workers to consume ready candidates without synchronously blocking on per-trial ask/suggest work.

#### Scenario: Fill candidate queue from batch primitive
- **WHEN** candidate queue mode is enabled
- **THEN** the coordinator fills the ready queue through batch reservation and batch suggestion rather than by repeatedly calling public `ask`

#### Scenario: Bound queue staleness
- **WHEN** the candidate queue contains ready candidates
- **THEN** each candidate records a snapshot identifier, queue age, batch identifier, and maximum allowed staleness

#### Scenario: Apply backpressure
- **WHEN** the candidate queue reaches its configured maximum depth or maximum inflight trial count
- **THEN** the coordinator stops prefetching new candidates until capacity is available

#### Scenario: Preserve worker hot-path throughput
- **WHEN** a worker requests a candidate and the ready queue is non-empty
- **THEN** the worker receives the candidate without performing storage trial creation or sampler suggestion on that worker request path

### Requirement: Explicit adaptiveness controls
Optuna SHALL expose configuration that makes throughput-versus-adaptiveness tradeoffs explicit for high-throughput ask/tell usage.

#### Scenario: Configure throughput knobs
- **WHEN** high-throughput mode is configured
- **THEN** the user can set batch size, maximum inflight trials, candidate queue size, maximum snapshot age, lease timeout, and generator mode

#### Scenario: Report tradeoff metadata
- **WHEN** high-throughput mode produces candidates
- **THEN** Optuna records metadata sufficient to explain the tradeoff used for each candidate batch

#### Scenario: Warn on low objective runtime
- **WHEN** objective runtime is low enough that optimizer coordination may dominate wall-clock throughput
- **THEN** Optuna documentation or reporting describes when batch, queue, or cheap-generator modes may outperform precise per-trial adaptiveness

### Requirement: Cheap generator mode
Optuna SHALL allow high-throughput users to select cheaper candidate-generation modes for workloads where best-value-per-second matters more than maximum per-sample adaptiveness.

#### Scenario: Select cheap generator
- **WHEN** a user selects Random, QMC, Sobol-style, or equivalent cheap generator mode for high-throughput optimization
- **THEN** Optuna uses that generator for candidate production and records the selected generator mode in trial or batch metadata

#### Scenario: Compare against adaptive sampler
- **WHEN** cheap generator mode is benchmarked or reported
- **THEN** Optuna compares throughput and optimization quality against the configured adaptive sampler baseline

### Requirement: Resource scheduling separation
Optuna SHALL keep heterogeneous worker or GPU resource scheduling separate from sampler scoring while allowing high-throughput candidate production to keep workers saturated.

#### Scenario: Assign ready candidates to heterogeneous workers
- **WHEN** basic and advanced workers or GPU profiles are available
- **THEN** the scheduler can assign ready candidates based on resource capacity without changing the candidate's sampler score

#### Scenario: Record resource metadata
- **WHEN** a candidate is assigned to a worker or device profile
- **THEN** Optuna records resource assignment metadata separately from sampler and trial-result metadata

### Requirement: Reproducibility metadata
Optuna SHALL persist enough metadata to reproduce or audit high-throughput candidate generation and result application order.

#### Scenario: Record batch provenance
- **WHEN** a batch of candidates is generated
- **THEN** Optuna records batch ID, reservation order, sampler or generator mode, RNG state or seed reference, pending set summary, and snapshot identifier

#### Scenario: Record result application order
- **WHEN** multiple trial results are applied asynchronously or through batch completion
- **THEN** Optuna records the order in which results were accepted into the study history

### Requirement: Backward compatibility
Optuna SHALL preserve existing ask/tell behavior unless high-throughput features are explicitly enabled.

#### Scenario: Default single-trial ask remains unchanged
- **WHEN** a user calls existing `Study.ask` without high-throughput mode
- **THEN** Optuna follows the existing single-trial ask behavior

#### Scenario: Default single-trial tell remains unchanged
- **WHEN** a user calls existing `Study.tell` without high-throughput mode
- **THEN** Optuna follows the existing single-trial tell behavior
