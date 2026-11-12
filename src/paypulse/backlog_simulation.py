
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.paypulse.regimes import (
    generate_independent_regimes,
    generate_markov_regimes,
)
from src.paypulse.risk_metrics import calculate_risk_metrics


@dataclass(frozen=True)
class BacklogSimulationConfig:
    initial_balance: float = 6000.0
    daily_expenses: float = 650.0

    payments_per_day: int = 10
    payment_amount: float = 60.0

    reserve_threshold: float = 5600.0
    days: int = 7
    scenarios: int = 5000

    normal_delay_probability: float = 0.05
    stress_delay_probability: float = 0.50

    normal_resolution_probability: float = 0.75
    stress_resolution_probability: float = 0.25

    refund_probability: float = 0.08
    minimum_refund_fraction: float = 0.10
    maximum_refund_fraction: float = 0.50

    regime_model: str = "markov"

    disruption_probability: float = 0.08
    normal_to_stress: float = 0.04
    stress_to_normal: float = 0.46

    seed: int = 42


def validate_backlog_config(config):
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

    probabilities = {
        "Normal delay": config.normal_delay_probability,
        "Stress delay": config.stress_delay_probability,
        "Normal resolution": config.normal_resolution_probability,
        "Stress resolution": config.stress_resolution_probability,
        "Disruption": config.disruption_probability,
        "Normal-to-stress": config.normal_to_stress,
        "Stress-to-normal": config.stress_to_normal,
        "refund_probability": config.refund_probability
    }

    for name, probability in probabilities.items():
        if not 0 <= probability <= 1:
            raise ValueError(f"{name} probability must be between 0 and 1.")

    if not (
    0 <= config.minimum_refund_fraction
    <= config.maximum_refund_fraction
    <= 1
    ):
        
        raise ValueError("Invalid refund fraction range.")

    if config.stress_delay_probability < config.normal_delay_probability:
        raise ValueError("Stress delay probability must be at least normal delay probability.")

    if config.stress_resolution_probability > config.normal_resolution_probability:
        raise ValueError("Stress resolution probability cannot exceed normal resolution probability.")

    if config.regime_model not in ("independent", "markov"):
        raise ValueError("Regime model must be 'independent' or 'markov'.")

    if (
        config.regime_model == "markov"
        and config.normal_to_stress + config.stress_to_normal == 0
    ):
        raise ValueError("At least one Markov transition probability must be positive.")


def run_backlog_simulation(config):
    validate_backlog_config(config)

    n = config.scenarios
    days = config.days

    # Generate daily disruption states.
    if config.regime_model == "independent":
        regimes = generate_independent_regimes(
            scenarios=n,
            days=days,
            disruption_probability=config.disruption_probability,
            seed=config.seed,
        )
    else:
        regimes = generate_markov_regimes(
            scenarios=n,
            days=days,
            normal_to_stress=config.normal_to_stress,
            stress_to_normal=config.stress_to_normal,
            seed=config.seed,
        )

    # Separate random generator for payment-level events.
    rng = np.random.default_rng(config.seed + 1)

    balances = np.zeros((n, days + 1))
    balances[:, 0] = config.initial_balance

    minimum_balances = np.full(n, config.initial_balance)

    # Each scenario maintains its own outstanding payment backlog.
    backlogs = [[] for _ in range(n)]
    pending_refunds = [[] for _ in range(n)]

    daily_rows = []
    event_rows = []

    for day in range(days):

        for scenario in range(n):

            stressed = bool(regimes[scenario, day])

            resolution_probability = (
                config.stress_resolution_probability
                if stressed
                else config.normal_resolution_probability
            )

            delay_probability = (
                config.stress_delay_probability
                if stressed
                else config.normal_delay_probability
            )

            captured_amount = 0.0

            refunded_amount = 0.0

            # Process refunds scheduled on previous days.
            todays_refunds = pending_refunds[scenario]
            pending_refunds[scenario] = []

            for refund in todays_refunds:

                refunded_amount += refund["amount"]

                event_rows.append({
                    "scenario": scenario,
                    "day": day + 1,
                    "order_id": refund["order_id"],
                    "event_type": "refund_processed",
                    "amount": refund["amount"],
                })

            
            # 1. Resolve some previously delayed payments.
            outstanding = backlogs[scenario]
            remaining = []

            for payment in outstanding:

                resolved = rng.random() < resolution_probability

                if resolved:

                    captured_amount += payment["amount"]

                    event_rows.append({
                        "scenario": scenario,
                        "day": day + 1,
                        "order_id": payment["order_id"],
                        "event_type": "delayed_capture_received",
                        "amount": payment["amount"],
                    })

                    # Refund eligibility begins after capture.
                    if rng.random() < config.refund_probability:

                        refund_fraction = rng.uniform(
                            config.minimum_refund_fraction,
                            config.maximum_refund_fraction,
                        )

                        refund_amount = payment["amount"] * refund_fraction

                        pending_refunds[scenario].append({
                            "order_id": payment["order_id"],
                            "amount": refund_amount,
                        })

                        event_rows.append({
                            "scenario": scenario,
                            "day": day + 1,
                            "order_id": payment["order_id"],
                            "event_type": "refund_scheduled",
                            "amount": refund_amount,
                        })

                else:
                    # Keep unresolved payments in the backlog.
                    remaining.append(payment)

            backlogs[scenario] = remaining


            # 2. Generate new payment orders.
            for payment_number in range(config.payments_per_day):

                order_id = (
                    f"S{scenario + 1}-D{day + 1}-P{payment_number + 1}"
                )

                amount = config.payment_amount

                delayed = rng.random() < delay_probability

                if delayed:

                    backlogs[scenario].append({
                        "order_id": order_id,
                        "amount": amount,
                    })

                    event_rows.append({
                        "scenario": scenario,
                        "day": day + 1,
                        "order_id": order_id,
                        "event_type": "payment_delayed",
                        "amount": amount,
                    })

                else:
                    captured_amount += amount

                    event_rows.append({
                        "scenario": scenario,
                        "day": day + 1,
                        "order_id": order_id,
                        "event_type": "payment_captured",
                        "amount": amount,
                    })

                    # Schedule a refund only after successful capture.
                    if rng.random() < config.refund_probability:

                        refund_fraction = rng.uniform(
                            config.minimum_refund_fraction,
                            config.maximum_refund_fraction,
                        )

                        refund_amount = amount * refund_fraction

                        pending_refunds[scenario].append({
                            "order_id": order_id,
                            "amount": refund_amount,
                        })

                        event_rows.append({
                        "scenario": scenario,
                        "day": day + 1,
                        "order_id": order_id,
                        "event_type": "refund_scheduled",
                        "amount": refund_amount,
                        })

            # 3. Update merchant liquidity.
                net_flow = (
                    captured_amount - refunded_amount - config.daily_expenses
                    )

            balances[scenario, day + 1] = (
                balances[scenario, day] + net_flow
            )

            minimum_balances[scenario] = min(
                minimum_balances[scenario],
                balances[scenario, day + 1],
            )

            outstanding_amount = sum(
                payment["amount"]
                for payment in backlogs[scenario]
            )

            daily_rows.append({
                "scenario": scenario,
                "day": day + 1,
                "balance": balances[scenario, day + 1],
                "stress_regime": stressed,
                "outstanding_payment_count": len(backlogs[scenario]),
                "outstanding_payment_amount": outstanding_amount,
            })

    # Final scenario-level results.
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
        "outstanding_payment_count": [
            len(backlog) for backlog in backlogs
        ],
        "outstanding_payment_amount": [
            sum(payment["amount"] for payment in backlog)
            for backlog in backlogs
        ],
    })

    daily = pd.DataFrame(daily_rows)
    events = pd.DataFrame(event_rows)

    return daily, results, events


def summarize_backlog_results(config, results):

    if results.empty:
        raise ValueError("Simulation results cannot be empty.")

    summary = calculate_risk_metrics(
        minimum_balances=results["minimum_balance"].to_numpy(),
        ending_balances=results["ending_balance"].to_numpy(),
        reserve_threshold=config.reserve_threshold,
    )

    summary.update({
        "average_outstanding_payment_count": float(
            results["outstanding_payment_count"].mean()
        ),
        "average_outstanding_payment_amount": float(
            results["outstanding_payment_amount"].mean()
        ),
    })

    return summary
