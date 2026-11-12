
import pytest

from src.paypulse.simulation import (
    SimulationConfig,
    run_simulation,
    summarize_results,
)


def test_simulation_is_reproducible():
    config = SimulationConfig(scenarios=100, seed=123)

    _, first = run_simulation(config)
    _, second = run_simulation(config)

    assert first.equals(second)


def test_simulation_has_expected_dimensions():
    config = SimulationConfig(scenarios=100, days=7)

    daily, results = run_simulation(config)

    assert daily.shape == (100, 8)
    assert len(results) == 100
    assert results["reserve_breached"].dtype == bool


def test_breach_probability_is_valid():
    config = SimulationConfig(scenarios=100)

    _, results = run_simulation(config)
    summary = summarize_results(config, results)

    assert 0 <= summary["reserve_breach_probability"] <= 1


def test_invalid_probability_is_rejected():
    config = SimulationConfig(delay_probability=1.5)

    with pytest.raises(ValueError):
        run_simulation(config)
