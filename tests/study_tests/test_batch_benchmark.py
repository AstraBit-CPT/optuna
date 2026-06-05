from __future__ import annotations

import math

import pytest

import optuna
from optuna import distributions
from optuna.exceptions import ExperimentalWarning
from optuna.study._batch_benchmark import BatchThroughputBenchmarkScenario
from optuna.study._batch_benchmark import run_batch_throughput_benchmark


def test_ask_batch_reports_coordination_timing_metadata() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    fixed_distributions = {"x": distributions.FloatDistribution(0, 1)}

    with pytest.warns(ExperimentalWarning):
        result = study.ask_batch(2, fixed_distributions=fixed_distributions)

    assert result.metadata.storage_time >= 0.0
    assert result.metadata.sampler_time >= 0.0


def test_run_batch_throughput_benchmark_reports_required_metrics() -> None:
    results = run_batch_throughput_benchmark(n_trials=4, batch_size=2, seed=3)

    assert [result.scenario for result in results] == list(
        BatchThroughputBenchmarkScenario
    )
    assert len(results) == 7

    results_by_scenario = {result.scenario: result for result in results}
    assert (
        "repeated_single_suggestion"
        in results_by_scenario[
            BatchThroughputBenchmarkScenario.ASK_BATCH_REPEATED_SUGGESTION
        ].fallback_modes
    )
    assert (
        "none"
        in results_by_scenario[
            BatchThroughputBenchmarkScenario.ASK_BATCH_CHEAP_RANDOM
        ].fallback_modes
    )

    for result in results:
        assert result.trial_count == 4
        assert result.elapsed_time > 0.0
        assert result.throughput > 0.0
        assert result.candidate_wait_p95 >= 0.0
        assert result.storage_time >= 0.0
        assert result.sampler_time >= 0.0
        assert 0.0 <= result.duplicate_rate <= 1.0
        assert math.isfinite(result.best_value)
        assert math.isfinite(result.best_value_per_second)
        assert math.isfinite(result.fixed_budget_regret)
        assert result.lease_reclaim_count == 0
        assert result.reproducibility_metadata["scenario"] == result.scenario.value
        assert result.reproducibility_metadata["n_trials"] == 4
        assert result.reproducibility_metadata["batch_size"] == 2
        assert result.reproducibility_metadata["seed"] == 3


def test_run_batch_throughput_benchmark_validates_configuration() -> None:
    with pytest.raises(ValueError):
        run_batch_throughput_benchmark(n_trials=0)

    with pytest.raises(ValueError):
        run_batch_throughput_benchmark(batch_size=0)

    with pytest.raises(ValueError):
        run_batch_throughput_benchmark(scenarios=["unknown"])
