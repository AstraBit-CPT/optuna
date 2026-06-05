from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
import random
import time
from typing import Any
import warnings

import optuna
from optuna.distributions import _get_single_value
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.study._batch import BatchFallbackMode
from optuna.study._batch import BatchGeneratorMode
from optuna.study._batch import BatchTellInput
from optuna.study._batch_queue import BatchQueueAcquireStatus


class BatchThroughputBenchmarkScenario(Enum):
    """Scenarios for high-throughput ask/tell benchmark smoke runs."""

    SYNC_ASK_TELL = "sync_ask_tell"
    ASK_BATCH_REPEATED_SUGGESTION = "ask_batch_repeated_suggestion"
    ASK_BATCH_NATIVE_RANDOM = "ask_batch_native_random"
    ASK_BATCH_TPE_CONSTANT_LIAR = "ask_batch_tpe_constant_liar"
    CANDIDATE_QUEUE_RANDOM = "candidate_queue_random"
    ASK_BATCH_CHEAP_RANDOM = "ask_batch_cheap_random"
    OPTUNA_AS_LEDGER = "optuna_as_ledger"


@dataclass(frozen=True)
class BatchThroughputBenchmarkResult:
    """Metrics emitted by one high-throughput ask/tell benchmark scenario."""

    scenario: BatchThroughputBenchmarkScenario
    trial_count: int
    elapsed_time: float
    throughput: float
    candidate_wait_p95: float
    storage_time: float
    sampler_time: float
    duplicate_rate: float
    best_value: float
    best_value_per_second: float
    fixed_budget_regret: float
    lease_reclaim_count: int
    fallback_modes: tuple[str, ...]
    reproducibility_metadata: dict[str, Any]


DEFAULT_BATCH_BENCHMARK_DISTRIBUTIONS: dict[str, BaseDistribution] = {
    "x": FloatDistribution(-5.0, 5.0),
    "y": FloatDistribution(-5.0, 5.0),
}


def run_batch_throughput_benchmark(
    *,
    scenarios: Sequence[BatchThroughputBenchmarkScenario | str] | None = None,
    n_trials: int = 16,
    batch_size: int = 4,
    seed: int = 0,
    fixed_distributions: Mapping[str, BaseDistribution] | None = None,
) -> list[BatchThroughputBenchmarkResult]:
    """Run a short benchmark comparing high-throughput ask/tell candidate paths.

    The helper is intentionally small enough for smoke validation. It records the metric surface
    needed by adoption gates without claiming that the default trial counts are statistically
    meaningful.
    """

    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    normalized_scenarios = [
        _normalize_scenario(scenario)
        for scenario in (
            scenarios
            if scenarios is not None
            else tuple(BatchThroughputBenchmarkScenario)
        )
    ]
    search_space = dict(fixed_distributions or DEFAULT_BATCH_BENCHMARK_DISTRIBUTIONS)

    return [
        _run_batch_throughput_scenario(
            scenario=scenario,
            n_trials=n_trials,
            batch_size=batch_size,
            seed=seed,
            fixed_distributions=search_space,
        )
        for scenario in normalized_scenarios
    ]


def _run_batch_throughput_scenario(
    *,
    scenario: BatchThroughputBenchmarkScenario,
    n_trials: int,
    batch_size: int,
    seed: int,
    fixed_distributions: dict[str, BaseDistribution],
) -> BatchThroughputBenchmarkResult:
    candidate_waits: list[float] = []
    storage_times: list[float] = []
    sampler_times: list[float] = []
    params_history: list[dict[str, Any]] = []
    values: list[float] = []
    fallback_modes: list[BatchFallbackMode] = []

    started_at = time.perf_counter()

    if scenario is BatchThroughputBenchmarkScenario.SYNC_ASK_TELL:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
        for _ in range(n_trials):
            wait_started_at = time.perf_counter()
            trial = study.ask(fixed_distributions=fixed_distributions)
            candidate_waits.append(time.perf_counter() - wait_started_at)
            value = _objective(trial.params)
            study.tell(trial, value)
            params_history.append(dict(trial.params))
            values.append(value)

    elif scenario is BatchThroughputBenchmarkScenario.ASK_BATCH_REPEATED_SUGGESTION:
        study = optuna.create_study(sampler=_FallbackRandomSampler(seed=seed))
        _run_ask_batch_loop(
            study=study,
            n_trials=n_trials,
            batch_size=batch_size,
            fixed_distributions=fixed_distributions,
            candidate_waits=candidate_waits,
            storage_times=storage_times,
            sampler_times=sampler_times,
            params_history=params_history,
            values=values,
            fallback_modes=fallback_modes,
        )

    elif scenario is BatchThroughputBenchmarkScenario.ASK_BATCH_NATIVE_RANDOM:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
        _run_ask_batch_loop(
            study=study,
            n_trials=n_trials,
            batch_size=batch_size,
            fixed_distributions=fixed_distributions,
            candidate_waits=candidate_waits,
            storage_times=storage_times,
            sampler_times=sampler_times,
            params_history=params_history,
            values=values,
            fallback_modes=fallback_modes,
        )

    elif scenario is BatchThroughputBenchmarkScenario.ASK_BATCH_TPE_CONSTANT_LIAR:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(
                constant_liar=True, n_startup_trials=0, seed=seed
            )
        study = optuna.create_study(sampler=sampler)
        _run_ask_batch_loop(
            study=study,
            n_trials=n_trials,
            batch_size=batch_size,
            fixed_distributions=fixed_distributions,
            candidate_waits=candidate_waits,
            storage_times=storage_times,
            sampler_times=sampler_times,
            params_history=params_history,
            values=values,
            fallback_modes=fallback_modes,
        )

    elif scenario is BatchThroughputBenchmarkScenario.CANDIDATE_QUEUE_RANDOM:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            queue = study.create_batch_candidate_queue(
                batch_size=batch_size,
                max_queue_size=batch_size,
                max_inflight=batch_size,
                fixed_distributions=fixed_distributions,
                generator_mode=BatchGeneratorMode.RANDOM,
                generator_seed=seed,
            )
        completed_count = 0
        while completed_count < n_trials:
            if queue.ready_count == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ExperimentalWarning)
                    refill = queue.refill()
                if refill.ask_metadata is not None:
                    storage_times.append(refill.ask_metadata.storage_time)
                    sampler_times.append(refill.ask_metadata.sampler_time)
                    fallback_modes.append(refill.ask_metadata.fallback_mode)

            wait_started_at = time.perf_counter()
            acquired = queue.acquire(
                "benchmark-worker", resource_profile={"kind": "cpu", "slots": 1}
            )
            candidate_waits.append(time.perf_counter() - wait_started_at)
            if acquired.status is not BatchQueueAcquireStatus.READY:
                continue
            assert acquired.trial_handle is not None
            value = _objective(acquired.trial_handle.trial.params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExperimentalWarning)
                queue.tell(acquired.trial_handle, values=value)
            params_history.append(dict(acquired.trial_handle.trial.params))
            values.append(value)
            completed_count += 1

    elif scenario is BatchThroughputBenchmarkScenario.ASK_BATCH_CHEAP_RANDOM:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
        _run_ask_batch_loop(
            study=study,
            n_trials=n_trials,
            batch_size=batch_size,
            fixed_distributions=fixed_distributions,
            candidate_waits=candidate_waits,
            storage_times=storage_times,
            sampler_times=sampler_times,
            params_history=params_history,
            values=values,
            fallback_modes=fallback_modes,
            generator_mode=BatchGeneratorMode.RANDOM,
            generator_seed=seed,
        )

    elif scenario is BatchThroughputBenchmarkScenario.OPTUNA_AS_LEDGER:
        study = optuna.create_study()
        rng = random.Random(seed)
        for _ in range(n_trials):
            wait_started_at = time.perf_counter()
            params = _sample_ledger_params(rng, fixed_distributions)
            candidate_waits.append(time.perf_counter() - wait_started_at)
            value = _objective(params)
            storage_started_at = time.perf_counter()
            study.add_trial(
                optuna.create_trial(
                    params=params,
                    distributions=fixed_distributions,
                    value=value,
                )
            )
            storage_times.append(time.perf_counter() - storage_started_at)
            sampler_times.append(candidate_waits[-1])
            params_history.append(params)
            values.append(value)

    elapsed_time = time.perf_counter() - started_at
    best_value = min(values) if values else math.nan

    return BatchThroughputBenchmarkResult(
        scenario=scenario,
        trial_count=len(values),
        elapsed_time=elapsed_time,
        throughput=(len(values) / elapsed_time if elapsed_time > 0 else math.inf),
        candidate_wait_p95=_percentile(candidate_waits, 0.95),
        storage_time=sum(storage_times),
        sampler_time=sum(sampler_times),
        duplicate_rate=_calculate_duplicate_rate(params_history),
        best_value=best_value,
        best_value_per_second=(best_value / elapsed_time if elapsed_time > 0 else math.nan),
        fixed_budget_regret=best_value,
        lease_reclaim_count=0,
        fallback_modes=tuple(dict.fromkeys(mode.value for mode in fallback_modes)),
        reproducibility_metadata={
            "scenario": scenario.value,
            "n_trials": n_trials,
            "batch_size": batch_size,
            "seed": seed,
            "search_space": sorted(fixed_distributions),
        },
    )


def _run_ask_batch_loop(
    *,
    study: optuna.Study,
    n_trials: int,
    batch_size: int,
    fixed_distributions: dict[str, BaseDistribution],
    candidate_waits: list[float],
    storage_times: list[float],
    sampler_times: list[float],
    params_history: list[dict[str, Any]],
    values: list[float],
    fallback_modes: list[BatchFallbackMode],
    generator_mode: BatchGeneratorMode | None = None,
    generator_seed: int | None = None,
) -> None:
    while len(values) < n_trials:
        count = min(batch_size, n_trials - len(values))
        wait_started_at = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            ask_result = study.ask_batch(
                count,
                fixed_distributions=fixed_distributions,
                generator_mode=generator_mode,
                generator_seed=generator_seed,
            )
        candidate_wait = time.perf_counter() - wait_started_at
        candidate_waits.extend([candidate_wait] * len(ask_result.trials))
        storage_times.append(ask_result.metadata.storage_time)
        sampler_times.append(ask_result.metadata.sampler_time)
        fallback_modes.append(ask_result.metadata.fallback_mode)

        completions = []
        for trial in ask_result.trials:
            value = _objective(trial.params)
            params_history.append(dict(trial.params))
            values.append(value)
            completions.append(BatchTellInput(trial=trial, values=value))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            study.tell_batch(completions)


class _FallbackRandomSampler(optuna.samplers.RandomSampler):
    def _supports_native_batch_sampling(self) -> bool:
        return False


def _objective(params: Mapping[str, Any]) -> float:
    return sum(float(value) ** 2 for value in params.values() if isinstance(value, Real))


def _sample_ledger_params(
    rng: random.Random, fixed_distributions: Mapping[str, BaseDistribution]
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, distribution in fixed_distributions.items():
        if distribution.single():
            params[name] = _get_single_value(distribution)
        elif isinstance(distribution, FloatDistribution):
            params[name] = rng.uniform(distribution.low, distribution.high)
        elif isinstance(distribution, IntDistribution):
            params[name] = rng.randint(distribution.low, distribution.high)
        elif isinstance(distribution, CategoricalDistribution):
            params[name] = rng.choice(distribution.choices)
        else:
            raise ValueError(
                f"Unsupported ledger benchmark distribution for parameter {name!r}."
            )
    return params


def _calculate_duplicate_rate(params_history: Sequence[Mapping[str, Any]]) -> float:
    pair_count = len(params_history) * (len(params_history) - 1) // 2
    if pair_count == 0:
        return 0.0

    fingerprints = [_fingerprint_params(params) for params in params_history]
    duplicate_count = 0
    for i, fingerprint in enumerate(fingerprints[:-1]):
        for other in fingerprints[i + 1 :]:
            if fingerprint == other:
                duplicate_count += 1
    return duplicate_count / pair_count


def _fingerprint_params(params: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, repr(value)) for name, value in params.items()))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * q) - 1))
    return sorted_values[index]


def _normalize_scenario(
    scenario: BatchThroughputBenchmarkScenario | str,
) -> BatchThroughputBenchmarkScenario:
    if isinstance(scenario, BatchThroughputBenchmarkScenario):
        return scenario
    if isinstance(scenario, str):
        return BatchThroughputBenchmarkScenario(scenario)
    raise TypeError("scenario must be a BatchThroughputBenchmarkScenario or string.")
