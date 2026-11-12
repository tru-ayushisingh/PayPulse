
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    initial_balance: float = 6000.0
    daily_expenses: float = 650.0
    payments_per_day: int = 10
    payment_amount: float = 60.0
    reserve_threshold: float = 5600.0
    days: int = 7
    scenarios: int = 5000
    delay_probability: float = 0.08
    refund_probability: float = 0.08
    minimum_refund: float = 100.0
    maximum_refund: float = 250.0
    seed: int = 42


def validate_config(config: SimulationConfig) -> None:
    if config.initial_balance < 0:
        raise ValueError("Initial balance cannot be negative.")

    if config.daily_expenses < 0:
        raise ValueError("Daily expenses cannot be negative.")

    if config.payments_per_day < 0:
        raise ValueError("Payments per day cannot be negative.")

    if config.payment_amount < 0:
        raise ValueError("Payment amount cannot be negative.")

    if config.days < 1 or config.scenarios < 1:
        raise ValueError("Days and scenarios must be positive.")

    for name, probability in [
        ("delay_probability", config.delay_probability),
        ("refund_probability", config.refund_probability),
    ]:
        if not 0 <= probability <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    if (
        config.minimum_refund < 0
        or config.maximum_refund < config.minimum_refund
    ):
        raise ValueError("Invalid refund amount range.")


def run_simulation(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run reproducible, synthetic merchant cash-flow scenarios."""

    validate_config(config)
    rng = np.random.default_rng(config.seed)

    # Each row represents one simulated business week.
    balances = np.empty((config.scenarios, config.days + 1))
    balances[:, 0] = config.initial_balance

    for day in range(config.days):
        # Simulate payment arrivals.
        payment_delays = (
            rng.random((config.scenarios, config.payments_per_day))
            < config.delay_probability
        )

        delayed_count = payment_delays.sum(axis=1)
        receipts = (
            config.payments_per_day - delayed_count
        ) * config.payment_amount

        # Payments delayed on the previous day arrive today.
        if day > 0:
            receipts += (
                previous_delayed_count * config.payment_amount
            )

        # Delays on the final day arrive after our simulation horizon.
        previous_delayed_count = delayed_count

        # Simulate whether a refund occurs and its amount.
        refund_occurs = (
            rng.random(config.scenarios) < config.refund_probability
        )

        refund_amount = rng.uniform(
            config.minimum_refund,
            config.maximum_refund,
            size=config.scenarios,
        ) * refund_occurs

        net_flow = receipts - config.daily_expenses - refund_amount

        balances[:, day + 1] = balances[:, day] + net_flow

    daily = pd.DataFrame(
        balances,
        columns=[f"day_{i}" for i in range(config.days + 1)],
    )

    # Exclude the initial balance when measuring the minimum
    # operating balance during the simulated horizon.
    minimum_balances = balances[:, 1:].min(axis=1)
    ending_balances = balances[:, -1]

    results = pd.DataFrame({
        "minimum_balance": minimum_balances,
        "ending_balance": ending_balances,
        "reserve_breached": (
            minimum_balances < config.reserve_threshold
        ),
    })

    return daily, results


def deterministic_baseline(
    config: SimulationConfig,
) -> pd.DataFrame:
    """
    Calculate the expected cash-flow path without random sampling.

    Assumptions:
    - Payments are either immediate or delayed by one day.
    - Delayed payments arrive the following day.
    - Final-day delays arrive outside the simulation horizon.
    - Expected refunds equal probability * average refund amount.
    """

    validate_config(config)

    expected_payment_value = (
        config.payments_per_day * config.payment_amount
    )

    expected_refund = (
        config.refund_probability
        * (config.minimum_refund + config.maximum_refund)
        / 2
    )

    balances = [config.initial_balance]

    for day in range(config.days):

        # On the first day, only non-delayed payments arrive.
        if day == 0:
            expected_receipts = (
                expected_payment_value
                * (1 - config.delay_probability)
            )
        else:
            # Previous-day delayed payments arrive along with
            # today's non-delayed payments.
            expected_receipts = expected_payment_value

        next_balance = (
            balances[-1]
            + expected_receipts
            - config.daily_expenses
            - expected_refund
        )

        balances.append(next_balance)

    return pd.DataFrame({
        "day": range(config.days + 1),
        "expected_balance": balances,
    })


def summarize_results(
    config: SimulationConfig,
    results: pd.DataFrame,
) -> dict[str, float]:
    """Calculate liquidity risk and tail-risk measures."""

    validate_config(config)

    if results.empty:
        raise ValueError("Results cannot be empty.")

    if results["minimum_balance"].isna().any():
        raise ValueError("Minimum balances cannot contain NaN.")

    if results["ending_balance"].isna().any():
        raise ValueError("Ending balances cannot contain NaN.")

    minimum_balances = results["minimum_balance"].to_numpy()

    ending_balances = results["ending_balance"].to_numpy()

    # Positive values represent reserve shortfalls.
    losses = np.maximum(
        0,
        config.reserve_threshold - minimum_balances,
    )

    confidence = 0.95

    var_95 = float(np.quantile(losses, confidence))

    # Empirical CVaR using the Rockafellar-Uryasev
    # representation, which handles ties at the VaR threshold.
    cvar_95 = float(
        var_95
        + np.maximum(losses - var_95, 0).mean()
        / (1 - confidence)
    )

    return {
        "scenario_count": int(len(results)),
        "reserve_breach_probability": float(
            (minimum_balances < config.reserve_threshold).mean()
        ),
        "average_minimum_balance": float(
            minimum_balances.mean()
        ),
        "fifth_percentile_ending_balance": float(
            np.quantile(ending_balances, 0.05)
        ),
        "average_ending_balance": float(
            ending_balances.mean()
        ),
        "liquidity_var_95": var_95,
        "liquidity_cvar_95": cvar_95,
        "average_reserve_shortfall": float(losses.mean()),
    }
