
import numpy as np

from src.paypulse.event_simulation import (
    EventSimulationConfig,
    run_event_simulation,
    summarize_event_results,
)


def test_no_delays_no_refunds():
    config = EventSimulationConfig(
        initial_balance=1000,
        daily_expenses=100,
        payments_per_day=2,
        payment_amount=100,
        reserve_threshold=1000,
        normal_delay_probability=0,
        stress_delay_probability=0,
        disruption_probability=0,
        refund_probability=0,
        days=2,
        scenarios=10,
    )

    daily, results, events = run_event_simulation(config)

    expected_balance = [1000, 1100, 1200]

    assert daily.iloc[0].tolist() == expected_balance
    assert np.all(results["reserve_breached"] == False)


def test_delayed_payment_is_captured_next_day():
    config = EventSimulationConfig(
        initial_balance=1000,
        daily_expenses=0,
        payments_per_day=1,
        payment_amount=100,
        normal_delay_probability=1,
        stress_delay_probability=1,
        disruption_probability=1,
        refund_probability=0,
        days=2,
        scenarios=1,
    )

    daily, results, events = run_event_simulation(config)

    assert daily.iloc[0].tolist() == [1000, 1000, 1100]

    assert "delayed_capture_received" in (
        events["event_type"].values
    )


def test_refund_only_after_payment_capture():
    config = EventSimulationConfig(
        initial_balance=1000,
        daily_expenses=0,
        payments_per_day=1,
        payment_amount=100,
        normal_delay_probability=1,
        stress_delay_probability=1,
        disruption_probability=1,
        refund_probability=1,
        minimum_refund_fraction=0.5,
        maximum_refund_fraction=0.5,
        days=3,
        scenarios=1,
    )

    daily, results, events = run_event_simulation(config)

    order_events = events[
        events["order_id"] == "S1-D1-P1"
    ]

    event_types = order_events["event_type"].tolist()

    assert event_types.index("payment_delayed") < (
        event_types.index("delayed_capture_received")
    )

    assert event_types.index("delayed_capture_received") < (
        event_types.index("refund_scheduled")
    )

    assert event_types.index("refund_scheduled") < (
        event_types.index("refund_processed")
    )

    assert daily.iloc[0].tolist() == [
    1000,
    1000,
    1100,
    1150,
]

def test_common_disruption_creates_correlated_delays():
    config = EventSimulationConfig(
        payments_per_day=10,
        normal_delay_probability=0,
        stress_delay_probability=1,
        disruption_probability=0.5,
        refund_probability=0,
        days=3,
        scenarios=100,
    )

    _, _, events = run_event_simulation(config)

    payments = events[
        events["event_type"].isin([
            "payment_delayed",
            "payment_captured",
        ])
    ]

    grouped = payments.groupby(["scenario", "day"])

    # Under these deliberately extreme parameters,
    # all payments in a scenario/day share the same state.
    for _, group in grouped:
        assert len(group) == 10
        assert group["event_type"].nunique() == 1


def test_risk_metrics_are_valid():
    config = EventSimulationConfig(
        scenarios=100,
        days=3,
    )

    _, results, _ = run_event_simulation(config)

    summary = summarize_event_results(config, results)

    assert 0 <= summary["reserve_breach_probability"] <= 1

    assert summary["liquidity_cvar_95"] >= (
        summary["liquidity_var_95"]
    )


def test_reproducibility():
    config = EventSimulationConfig(
        scenarios=100,
        seed=123,
    )

    first = run_event_simulation(config)
    second = run_event_simulation(config)

    assert first[0].equals(second[0])
    assert first[1].equals(second[1])
    assert first[2].equals(second[2])
