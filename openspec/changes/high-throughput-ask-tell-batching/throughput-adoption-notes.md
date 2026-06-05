# High-Throughput Ask/Tell Adoption Notes

Last verified against public references on 2026-06-05.

## Premise

The goal is not "batching" by itself. The goal is to keep candidates available for millisecond-to-subsecond objectives when optimizer coordination can dominate wall-clock time. A loop that repeatedly calls `Study.ask()` preserves correctness, but it does not amortize storage reservation, sampler-history work, or parameter writes.

Use high-throughput mode only when measurement shows candidate acquisition is material compared with objective runtime. Keep ordinary `Study.ask()` / `Study.tell()` as the default precise-per-trial path.

## Mode Selection

| Mode | Use When | Main Tradeoff |
| --- | --- | --- |
| Synchronous ask/tell | Objective runtime is much larger than optimizer coordination, or maximum adaptiveness per trial matters most. | Lowest coordination complexity, weakest throughput ceiling for tiny objectives. |
| `ask_batch` with fallback labels | You need API compatibility and correctness before a backend or sampler has native support. | It is not a true throughput fix; fallback mode must be reported. |
| Native batch reservation and suggestion | Storage and sampler cost both matter, and the sampler can account for pending in-batch candidates. | Higher throughput, but suggestions are produced from a bounded snapshot. |
| Candidate queue | Worker hot-path latency matters more than immediate sampler freshness. | Queue size and max snapshot age must bound stale open-loop behavior. |
| Cheap generator mode | Random-style candidates can improve best-value-per-second because objectives are extremely cheap. | Less adaptive per sample; must be benchmarked against adaptive baselines. |
| Optuna-as-ledger control | You want to measure objective/executor overhead without Optuna candidate generation. | Useful as a lower-bound control, not an adaptive optimizer. |

## Required Metrics

Every adoption report should include:

- p95 candidate wait time.
- storage coordination time.
- sampler or generator time.
- completed trials per second.
- duplicate suggestion rate.
- best value per second.
- fixed-budget regret.
- lease reclaim count.
- fallback mode.
- reproducibility metadata: seed, search-space identifier, scenario, batch size, and trial budget.

The smoke fixture in `optuna.study._batch_benchmark.run_batch_throughput_benchmark` emits those fields for synchronous ask/tell, repeated-suggestion fallback, native batch reservation/suggestion, pending-aware TPE with `constant_liar`, candidate queue mode, cheap random generation, and Optuna-as-ledger control.

## Throughput Versus Adaptiveness

Batching and queues relax how fresh the sampler's view is. The right knob depends on objective runtime:

- If objective runtime is long, keep batch size and queue size small because sampler adaptiveness is cheap relative to evaluation.
- If objective runtime is short, increase batch size until p95 candidate wait stops dominating wall time, then watch duplicate rate and fixed-budget regret.
- If workers are heterogeneous, assign ready candidates by resource profile after candidate generation. Do not feed GPU capacity into sampler scoring unless the resource itself is part of the objective.
- If cheap generator mode beats adaptive BO on best-value-per-second, report that as a workload-specific result, not as a universal replacement for BO.

## Prior-Art Map

| System | Pattern To Learn From | Optuna Implication |
| --- | --- | --- |
| [Ray Tune ConcurrencyLimiter](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.ConcurrencyLimiter.html) | Concurrency is an explicit wrapper around a search algorithm. | Keep max inflight explicit and measurable instead of hiding it inside sampler behavior. |
| [Syne Tune](https://syne-tune.readthedocs.io/en/latest/index.html) | Scheduler, searcher, workers, and benchmark simulation are distinct concepts. | Keep candidate generation, execution scheduling, and benchmark gates separate. |
| [Ax trials and BatchTrial guidance](https://ax.dev/docs/1.1.2/experiment/) | Trial identity remains central even when evaluations run concurrently. | Preserve durable Optuna trial numbers and IDs before workers receive candidates. |
| [Google Vizier](https://research.google/pubs/google-vizier-a-service-for-black-box-optimization/) | Service-style black-box optimization emphasizes reliability and fault tolerance. | Leases, idempotent completion, and restart recovery are safety semantics around throughput. |
| [SMAC3 facade/intensifier APIs](https://automl.github.io/SMAC3/latest/api/smac/facade/hyperparameter_optimization_facade/) | Optimization strategy is separated from intensification/resource decisions. | Keep resource scheduling out of sampler scoring and record assignment metadata separately. |
| [OpenBox parallel evaluation](https://open-box.readthedocs.io/en/latest/advanced_usage/parallel_evaluation.html) | Sync and async parallel settings are explicit; parallel BO must avoid similar pending configurations. | Preserve pending-aware sampling and duplicate/near-duplicate diagnostics. |
| [Hyperopt SparkTrials](https://hyperopt.github.io/hyperopt/scaleout/spark/) | Parallelism trades scalability against adaptiveness; full parallelism degenerates toward random search. | Report fallback/generator mode and compare best-value-per-second against adaptive baselines. |
| [Nevergrad ask/tell docs](https://github.com/enthought/nevergrad/blob/master/docs/optimization.md#ask-and-tell-interface) | Ask/tell can be asynchronous and executor-agnostic. | Keep Optuna's scheduler boundary executor-neutral while adding batch and queue primitives. |

## Promotion Checklist

Do not promote this beyond experimental status until:

- Focused unit tests for batch reservation, pending-aware sampling, idempotent completion, leases, queue recovery, generator metadata, resource metadata, and benchmark metrics pass.
- A short benchmark smoke run emits all required scenarios and metric fields.
- At least one workload-specific benchmark uses enough trials to compare throughput and optimization quality, not just API behavior.
- Fallback paths are labeled as correctness paths rather than throughput paths.
- Documentation states the throughput-versus-adaptiveness tradeoff and points users to the benchmark fixture.
