## 1. API Contract and Capability Detection

- [x] 1.1 Decide the public API names for batch reservation and completion, including whether the main entrypoint is `ask_batch`, `reserve_trials`, or a coordinator object.
- [x] 1.2 Define typed result objects for reserved trials, batch provenance, lease tokens, and batch completion outcomes.
- [x] 1.3 Add storage capability detection for native batch reservation versus correctness-only fallback behavior.
- [x] 1.4 Add sampler capability detection for native batch suggestion versus repeated single-suggestion fallback behavior.
- [x] 1.5 Document backward compatibility expectations for existing `Study.ask` and `Study.tell`.

## 2. Batch Trial Reservation

- [x] 2.1 Add internal storage contract support for reserving multiple new trials in one operation.
- [x] 2.2 Implement RDB-backed batch reservation with unique trial IDs and trial numbers allocated under one storage coordination round.
- [x] 2.3 Add conservative fallback reservation behavior for storage backends without native batch support and label it as non-high-throughput.
- [x] 2.4 Ensure reserved trial identities are durable before workers receive them.
- [ ] 2.5 Add tests for unique trial IDs and trial numbers under concurrent batch reservation.

## 3. Pending-Aware Batch Suggestion

- [x] 3.1 Add a sampler extension point for generating suggestions for multiple reserved trials from one history/model pass.
- [x] 3.2 Implement default compatibility behavior for samplers without native batch support.
- [x] 3.3 Implement native pending-aware TPE batch suggestion that preserves `constant_liar` RUNNING-trial semantics.
- [x] 3.4 Ensure later in-batch suggestions account for earlier in-batch pending candidates.
- [x] 3.5 Add duplicate and near-duplicate metrics for batched sampler output.
- [x] 3.6 Add sampler tests comparing native batch behavior with repeated single-suggestion fallback behavior.

## 4. Batch Completion and Idempotency

- [x] 4.1 Add batch completion plumbing that records per-trial values, states, errors, and completion timestamps.
- [x] 4.2 Preserve existing single-trial `tell` behavior and compatibility paths.
- [x] 4.3 Add idempotency keys or completion tokens for reserved trials.
- [x] 4.4 Reject or ignore duplicate completion attempts after one completion has been accepted.
- [x] 4.5 Add tests for mixed successful, failed, and pruned outcomes in one batch completion request.

## 5. Lease Recovery

- [x] 5.1 Add lease metadata for pre-reserved candidates, including owner, token, deadline, and renewal count.
- [x] 5.2 Add lease renewal and expiration logic with configurable timeout.
- [x] 5.3 Add reclaim behavior for expired leased candidates.
- [x] 5.4 Add restart recovery logic that can reconstruct inflight leased work from durable state.
- [x] 5.5 Add tests for worker crash, coordinator restart, expired lease reclaim, and duplicate completion.

## 6. Bounded Candidate Queue

- [x] 6.1 Add an opt-in coordinator that fills a bounded ready queue from the batch primitive.
- [x] 6.2 Enforce candidate queue size, maximum inflight trial count, maximum snapshot age, and backpressure.
- [x] 6.3 Ensure worker dequeue from a non-empty ready queue does not perform storage trial creation or sampler suggestion on the worker request path.
- [x] 6.4 Persist or report batch ID, snapshot identifier, queue age, and reservation order for queued candidates.
- [x] 6.5 Add tests proving queue mode avoids repeated public `ask` calls while the ready queue is non-empty.

## 7. Cheap Generator Modes and Resource Scheduling

- [ ] 7.1 Add high-throughput generator controls for Random, QMC, Sobol-style, or equivalent cheap candidate generation where supported.
- [ ] 7.2 Record generator mode in batch and trial metadata.
- [ ] 7.3 Keep heterogeneous worker or GPU resource scheduling separate from sampler scoring.
- [ ] 7.4 Record worker/device assignment metadata separately from sampler and trial-result metadata.
- [ ] 7.5 Add tests or benchmark fixtures comparing cheap generator modes against adaptive sampler baselines on low-duration objectives.

## 8. Metrics, Documentation, and Benchmark Gates

- [ ] 8.1 Add benchmark scenarios comparing synchronous ask/tell, repeated-ask batch fallback, true batch reservation, pending-aware batch suggestion, candidate queue mode, cheap generator mode, and Optuna-as-ledger control behavior where practical.
- [ ] 8.2 Record p95 candidate wait, storage time, sampler time, throughput, duplicate rate, best-value-per-second, fixed-budget regret, lease reclaim count, and reproducibility metadata.
- [ ] 8.3 Add documentation explaining throughput-versus-adaptiveness tradeoffs and when cheap generators may beat adaptive BO for millisecond objectives.
- [ ] 8.4 Add documentation mapping comparable patterns from Ray Tune, Syne Tune, Ax, Vizier, SMAC, OpenBox, Hyperopt SparkTrials, and Nevergrad.
- [ ] 8.5 Run the focused unit tests and a short benchmark smoke run before promoting the feature beyond experimental status.
