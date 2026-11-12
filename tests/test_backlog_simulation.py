
import pytest

from src.paypulse.backlog_simulation import (
    BacklogSimulationConfig,
    run_backlog_simulation,
    summarize_backlog_results,
)


def test_no_delays_produces_no_backlog():

    config = BacklogSimulationConfig(
        initial_balance=1000,
        daily_expenses=100,
        payments_per_day=2,
        payment_amount=100,
        normal_delay_probability=0,
        stress_delay_probability=0,
        disruption_probability=0,
        days=2,
        scenarios=5,
    )

    daily, results, events = run_backlog_simulation(config)

    assert results["outstanding_payment_count"].sum() == 0
    assert results["outstanding_payment_amount"].sum() == 0

    assert daily.iloc[-1]["balance"] == 1200


def test_delayed_payments_remain_outstanding():

    config = BacklogSimulationConfig(
        initial_balance=1000,
        daily_expenses=0,
        payments_per_day=1,
        payment_amount=100,
        normal_delay_probability=1,
        stress_delay_probability=1,
        normal_resolution_probability=0,
        stress_resolution_probability=0,
        disruption_probability=0,
        days=3,
        scenarios=1,
    )

    daily, results, events = run_backlog_simulation(config)

    assert results.iloc[0]["outstanding_payment_count"] == 3
    assert results.iloc[0]["outstanding_payment_amount"] == 300

    assert daily.iloc[-1]["balance"] == 1000


def test_delayed_payments_can_resolve():

    config = BacklogSimulationConfig(
        initial_balance=1000,
        daily_expenses=0,
        payments_per_day=1,
        payment_amount=100,
        normal_delay_probability=1,
        stress_delay_probability=1,
        normal_resolution_probability=1,
        stress_resolution_probability=1,
        disruption_probability=0,
        days=2,
        scenarios=1,
    )

    daily, results, events = run_backlog_simulation(config)

    assert results.iloc[0]["outstanding_payment_count"] == 1
    assert results.iloc[0]["outstanding_payment_amount"] == 100

    assert "delayed_capture_received" in events["event_type"].values


def test_markov_and_independent_models_run():

    for model in ("independent", "markov"):

        config = BacklogSimulationConfig(
            regime_model=model,
            scenarios=20,
            days=5,
        )

        daily, results, events = run_backlog_simulation(config)

        assert len(results) == 20
        assert len(daily) == 100


def test_reproducibility():

    config = BacklogSimulationConfig(
        scenarios=20,
        days=5,
        seed=123,
    )

    first = run_backlog_simulation(config)
    second = run_backlog_simulation(config)

    assert first[0].equals(second[0])
    assert first[1].equals(second[1])
    assert first[2].equals(second[2])


def test_invalid_regime_model_rejected():

    config = BacklogSimulationConfig(
        regime_model="unknown",
    )

    with pytest.raises(ValueError):
        run_backlog_simulation(config)


def test_summary_uses_centralized_risk_metrics():

    config = BacklogSimulationConfig(
        scenarios=100,
        days=5,
    )

    _, results, _ = run_backlog_simulation(config)

    summary = summarize_backlog_results(config, results)

    assert 0 <= summary["reserve_breach_probability"] <= 1
    assert summary["liquidity_cvar_95"] >= summary["liquidity_var_95"]
    assert summary["average_outstanding_payment_amount"] >= 0


def test_refund_is_processed_after_capture():

    config = BacklogSimulationConfig(
        initial_balance=1000,
        daily_expenses=0,
        payments_per_day=1,
        payment_amount=100,
        normal_delay_probability=0,
        stress_delay_probability=0,
        refund_probability=1,
        minimum_refund_fraction=0.5,
        maximum_refund_fraction=0.5,
        days=2,
        scenarios=1,
    )

    daily, results, events = run_backlog_simulation(config)

    order_events = events[
        events["order_id"] == "S1-D1-P1"
    ]["event_type"].tolist()

    assert order_events.index("payment_captured") < (
        order_events.index("refund_scheduled")
    )

    assert order_events.index("refund_scheduled") < (
        order_events.index("refund_processed")
    )

    assert daily.iloc[-1]["balance"] == 1150


def test_delayed_payment_cannot_be_refunded_before_capture():

    config = BacklogSimulationConfig(
        initial_balance=1000,
        daily_expenses=0,
        payments_per_day=1,
        payment_amount=100,
        normal_delay_probability=1,
        stress_delay_probability=1,
        normal_resolution_probability=1,
        stress_resolution_probability=1,
        refund_probability=1,
        minimum_refund_fraction=0.5,
        maximum_refund_fraction=0.5,
        days=3,
        scenarios=1,
    )

    daily, results, events = run_backlog_simulation(config)

    order_events = events[
        events["order_id"] == "S1-D1-P1"
    ]["event_type"].tolist()

    assert order_events.index("payment_delayed") < (
        order_events.index("delayed_capture_received")
    )

    assert order_events.index("delayed_capture_received") < (
        order_events.index("refund_scheduled")
    )

    assert order_events.index("refund_scheduled") < (
        order_events.index("refund_processed")
    )


def test_invalid_refund_fraction_rejected():

    config = BacklogSimulationConfig(
        minimum_refund_fraction=0.8,
        maximum_refund_fraction=0.5,
    )

    import pytest

    with pytest.raises(ValueError):
        run_backlog_simulation(config)
