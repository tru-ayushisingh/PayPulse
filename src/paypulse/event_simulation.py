
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventSimulationConfig:
    initial_balance: float = 6000.0
    daily_expenses: float = 650.0

    payments_per_day: int = 10
    payment_amount: float = 60.0

    reserve_threshold: float = 5600.0
    days: int = 7
    scenarios: int = 5000

    normal_delay_probability: float = 0.05
    stress_delay_probability: float = 0.50
    disruption_probability: float = 0.08

    refund_probability: float = 0.08
    minimum_refund_fraction: float = 0.10
    maximum_refund_fraction: float = 0.50

    seed: int = 42


def validate_event_config(config):
    if config.initial_balance < 0:
        raise ValueError("Initial balance cannot be negative.")

    if config.daily_expenses < 0:
        raise ValueError("Expenses cannot be negative.")

    if config.payment_amount < 0:
        raise ValueError("Payment amount cannot be negative.")

    if config.payments_per_day < 0:
        raise ValueError("Payment count cannot be negative.")

    if config.days < 1 or config.scenarios < 1:
        raise ValueError("Days and scenarios must be positive.")

    probabilities = [
        config.normal_delay_probability,
        config.stress_delay_probability,
        config.disruption_probability,
        config.refund_probability,
    ]

    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be between 0 and 1.")

    if config.stress_delay_probability < config.normal_delay_probability:
        raise ValueError(
            "Stress delay probability must be at least normal delay probability."
        )

    if not (
        0 <= config.minimum_refund_fraction
        <= config.maximum_refund_fraction
        <= 1
    ):
        raise ValueError("Invalid refund fraction range.")


def run_event_simulation(config):
    validate_event_config(config)

    rng = np.random.default_rng(config.seed)

    n = config.scenarios
    days = config.days

    balances = np.zeros((n, days + 1))
    balances[:, 0] = config.initial_balance

    pending_captures = [[] for _ in range(n)]
    pending_refunds = [[] for _ in range(n)]

    event_rows = []

    for day in range(1, days + 1):

        for scenario in range(n):

            # Collect delayed captures and refunds due today.
            todays_captures = pending_captures[scenario]
            todays_refunds = pending_refunds[scenario]

            pending_captures[scenario] = []
            pending_refunds[scenario] = []

            captured_amount = 0.0
            refunded_amount = 0.0

            # Process previously delayed payments.
            for order_id, amount in todays_captures:

                captured_amount += amount

                event_rows.append({
                    "scenario": scenario,
                    "day": day,
                    "order_id": order_id,
                    "event_type": "delayed_capture_received",
                    "amount": amount,
                })

                # Delayed captures can also be refunded.
                if rng.random() < config.refund_probability:

                    fraction = rng.uniform(
                        config.minimum_refund_fraction,
                        config.maximum_refund_fraction,
                    )

                    refund = amount * fraction

                    pending_refunds[scenario].append(
                        (order_id, refund)
                    )

                    event_rows.append({
                        "scenario": scenario,
                        "day": day,
                        "order_id": order_id,
                        "event_type": "refund_scheduled",
                        "amount": refund,
                    })

            # Process refunds scheduled on previous days.
            for order_id, amount in todays_refunds:

                refunded_amount += amount

                event_rows.append({
                    "scenario": scenario,
                    "day": day,
                    "order_id": order_id,
                    "event_type": "refund_processed",
                    "amount": amount,
                })

            # Common disruption state for this scenario/day.
            disrupted = (
                rng.random() < config.disruption_probability
            )

            probability = (
                config.stress_delay_probability
                if disrupted
                else config.normal_delay_probability
            )

            # Generate today's new payment orders.
            for payment in range(config.payments_per_day):

                order_id = (
                    f"S{scenario + 1}-D{day}-P{payment + 1}"
                )

                amount = config.payment_amount

                delayed = rng.random() < probability

                if delayed:

                    pending_captures[scenario].append(
                        (order_id, amount)
                    )

                    event_rows.append({
                        "scenario": scenario,
                        "day": day,
                        "order_id": order_id,
                        "event_type": "payment_delayed",
                        "amount": amount,
                    })

                else:

                    captured_amount += amount

                    event_rows.append({
                        "scenario": scenario,
                        "day": day,
                        "order_id": order_id,
                        "event_type": "payment_captured",
                        "amount": amount,
                    })

                    if rng.random() < config.refund_probability:

                        fraction = rng.uniform(
                            config.minimum_refund_fraction,
                            config.maximum_refund_fraction,
                        )

                        refund = amount * fraction

                        pending_refunds[scenario].append(
                            (order_id, refund)
                        )

                        event_rows.append({
                            "scenario": scenario,
                            "day": day,
                            "order_id": order_id,
                            "event_type": "refund_scheduled",
                            "amount": refund,
                        })

            net_flow = (
                captured_amount
                - refunded_amount
                - config.daily_expenses
            )

            balances[scenario, day] = (
                balances[scenario, day - 1] + net_flow
            )

    daily = pd.DataFrame(
        balances,
        columns=[f"day_{i}" for i in range(days + 1)],
    )

    minimum_balances = balances[:, 1:].min(axis=1)

    results = pd.DataFrame({
        "minimum_balance": minimum_balances,
        "ending_balance": balances[:, -1],
        "reserve_breached": (
            minimum_balances < config.reserve_threshold
        ),
        "reserve_shortfall": np.maximum(
            0,
            config.reserve_threshold - minimum_balances,
        ),
    })

    events = pd.DataFrame(event_rows)

    return daily, results, events



def summarize_event_results(config, results):

    if results.empty:
        raise ValueError("Simulation results cannot be empty.")

    losses = results["reserve_shortfall"].to_numpy()

    var_95 = float(np.quantile(losses, 0.95))

    cvar_95 = float(
        var_95
        + np.maximum(losses - var_95, 0).mean() / 0.05
    )

    return {
        "scenario_count": len(results),

        "reserve_breach_probability": float(
            results["reserve_breached"].mean()
        ),

        "average_minimum_balance": float(
            results["minimum_balance"].mean()
        ),

        "average_ending_balance": float(
            results["ending_balance"].mean()
        ),

        "liquidity_var_95": var_95,

        "liquidity_cvar_95": cvar_95,

        "average_reserve_shortfall": float(
            losses.mean()
        ),
    }
