
from dataclasses import replace

import pandas as pd

from src.paypulse.simulation import (
    SimulationConfig,
    deterministic_baseline,
    run_simulation,
    summarize_results,
)


def run_sensitivity_analysis(
    config: SimulationConfig,
) -> pd.DataFrame:
    """
    One-factor-at-a-time sensitivity analysis.

    All other parameters remain fixed while one
    parameter is varied.
    """

    experiments = {
        "delay_probability": [0, 0.05, 0.10, 0.20, 0.30],
        "refund_probability": [0, 0.05, 0.10, 0.20, 0.30],
        "initial_balance": [4000, 5000, 6000, 7000, 8000],
    }

    rows = []

    for parameter, values in experiments.items():

        for value in values:

            experiment_config = replace(
                config,
                **{parameter: value},
            )

            _, results = run_simulation(experiment_config)

            summary = summarize_results(
                experiment_config,
                results,
            )

            rows.append({
                "parameter": parameter,
                "value": value,
                "breach_probability": summary[
                    "reserve_breach_probability"
                ],
                "var_95": summary["liquidity_var_95"],
                "cvar_95": summary["liquidity_cvar_95"],
                "average_ending_balance": summary[
                    "average_ending_balance"
                ],
            })

    return pd.DataFrame(rows)


def run_quantitative_research(
    config: SimulationConfig,
) -> dict:

    # Deterministic reference path.
    baseline = deterministic_baseline(config)

    # Monte Carlo experiment.
    daily, results = run_simulation(config)

    summary = summarize_results(config, results)

    # Sensitivity experiments.
    sensitivity = run_sensitivity_analysis(config)

    return {
        "deterministic_baseline": baseline,
        "simulation_summary": summary,
        "simulation_results": results,
        "daily_scenarios": daily,
        "sensitivity_analysis": sensitivity,
    }
