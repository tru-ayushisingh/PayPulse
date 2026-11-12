
import numpy as np

from src.paypulse.simulation import (
    SimulationConfig,
    deterministic_baseline,
    run_simulation,
    summarize_results,
)

from src.paypulse.research import run_sensitivity_analysis


def test_deterministic_baseline_without_delays_or_refunds():
    config = SimulationConfig(
        initial_balance=1000,
        daily_expenses=100,
        payments_per_day=10,
        payment_amount=60,
        delay_probability=0,
        refund_probability=0,
        days=2,
    )

    baseline = deterministic_baseline(config)

    assert baseline["expected_balance"].tolist() == [
        1000,
        1500,
        2000,
    ]


def test_deterministic_baseline_has_correct_dimensions():
    config = SimulationConfig(days=7)

    baseline = deterministic_baseline(config)

    assert len(baseline) == 8
    assert baseline.iloc[0]["day"] == 0
    assert baseline.iloc[-1]["day"] == 7


def test_cvar_is_not_less_than_var():
    config = SimulationConfig(
        scenarios=5000,
        seed=42,
    )

    _, results = run_simulation(config)

    summary = summarize_results(config, results)

    assert summary["liquidity_var_95"] >= 0

    assert (
        summary["liquidity_cvar_95"]
        >= summary["liquidity_var_95"]
    )


def test_reserve_shortfall_is_nonnegative():
    config = SimulationConfig(scenarios=1000)

    _, results = run_simulation(config)

    summary = summarize_results(config, results)

    assert summary["average_reserve_shortfall"] >= 0


def test_sensitivity_analysis_returns_all_experiments():
    config = SimulationConfig(
        scenarios=100,
        days=3,
    )

    results = run_sensitivity_analysis(config)

    assert len(results) == 15

    assert set(results["parameter"]) == {
        "delay_probability",
        "refund_probability",
        "initial_balance",
    }


def test_sensitivity_has_valid_probabilities():
    config = SimulationConfig(scenarios=100)

    results = run_sensitivity_analysis(config)

    assert np.all(
        results["breach_probability"].between(0, 1)
    )
