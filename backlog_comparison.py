
import numpy as np
import pandas as pd

from src.paypulse.backlog_simulation import (
    BacklogSimulationConfig,
    run_backlog_simulation,
    summarize_backlog_results,
)


# --------------------------------------------------
# 1. Experiment configuration
# --------------------------------------------------

COMMON_CONFIG = {
    "initial_balance": 6000,
    "daily_expenses": 650,
    "payments_per_day": 10,
    "payment_amount": 60,
    "reserve_threshold": 5500,
    "days": 7,
    "scenarios": 20000,
    "normal_delay_probability": 0.05,
    "stress_delay_probability": 0.50,
    "normal_resolution_probability": 0.75,
    "stress_resolution_probability": 0.25,
    "disruption_probability": 0.08,
    "normal_to_stress": 0.04,
    "stress_to_normal": 0.46,
    "seed": 42,
}

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 2026
CONFIDENCE_LEVEL = 0.95


# --------------------------------------------------
# 2. Run both simulation models
# --------------------------------------------------

model_results = {}
summaries = []

for model in ["independent", "markov"]:

    config = BacklogSimulationConfig(
        **COMMON_CONFIG,
        regime_model=model,
    )

    daily, results, events = run_backlog_simulation(config)

    model_results[model] = results.copy()

    summary = summarize_backlog_results(config, results)
    summary["regime_model"] = model

    summaries.append(summary)

    # Save raw scenario-level results.
    results.to_csv(
        f"backlog_{model}_results.csv",
        index=False,
    )

    print(f"\n===== {model.upper()} MODEL =====")
    print(pd.Series(summary).to_string())


# --------------------------------------------------
# 3. Calculate empirical CVaR
# --------------------------------------------------

def calculate_cvar(losses, confidence_level=0.95):

    var = np.quantile(losses, confidence_level)

    return float(
        var
        + np.maximum(losses - var, 0).mean()
        / (1 - confidence_level)
    )


# --------------------------------------------------
# 4. Define metrics for comparison
# --------------------------------------------------

def extract_metrics(results):

    return {
        "breach_probability": (
            results["reserve_breached"].to_numpy(dtype=float)
        ),

        "average_shortfall": (
            results["reserve_shortfall"].to_numpy(dtype=float)
        ),

        "cvar_95": (
            results["reserve_shortfall"].to_numpy(dtype=float)
        ),

        "outstanding_amount": (
            results["outstanding_payment_amount"].to_numpy(dtype=float)
        ),
    }


independent = extract_metrics(model_results["independent"])
markov = extract_metrics(model_results["markov"])


def calculate_statistic(values, metric):

    if metric == "breach_probability":
        return float(np.mean(values))

    if metric == "average_shortfall":
        return float(np.mean(values))

    if metric == "cvar_95":
        return calculate_cvar(values, CONFIDENCE_LEVEL)

    if metric == "outstanding_amount":
        return float(np.mean(values))

    raise ValueError(f"Unknown metric: {metric}")


# --------------------------------------------------
# 5. Bootstrap confidence intervals
# --------------------------------------------------


def bootstrap_difference(
    independent_values,
    markov_values,
    metric,
    iterations=2000,
    seed=2026,
):

    rng = np.random.default_rng(seed)

    independent_values = np.asarray(independent_values)
    markov_values = np.asarray(markov_values)

    if len(independent_values) != len(markov_values):
        raise ValueError("Both models must have matching scenario counts.")

    sample_count = len(independent_values)

    point_difference = (
        calculate_statistic(markov_values, metric)
        - calculate_statistic(independent_values, metric)
    )

    bootstrap_differences = np.empty(iterations)

    for i in range(iterations):

        # Resample the SAME scenario indices for both models.
        indices = rng.integers(
            0,
            sample_count,
            size=sample_count,
        )

        independent_sample = independent_values[indices]
        markov_sample = markov_values[indices]

        independent_stat = calculate_statistic(
            independent_sample,
            metric,
        )

        markov_stat = calculate_statistic(
            markov_sample,
            metric,
        )

        bootstrap_differences[i] = (
            markov_stat - independent_stat
        )

    lower = np.quantile(bootstrap_differences, 0.025)
    upper = np.quantile(bootstrap_differences, 0.975)

    return {
        "metric": metric,
        "markov_minus_independent": point_difference,
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "bootstrap_iterations": iterations,
    }


# --------------------------------------------------
# 6. Run all four comparisons
# --------------------------------------------------

comparisons = []

metrics_to_compare = [
    "breach_probability",
    "average_shortfall",
    "cvar_95",
    "outstanding_amount",
]

for index, metric in enumerate(metrics_to_compare):

    result = bootstrap_difference(
        independent_values=independent[metric],
        markov_values=markov[metric],
        metric=metric,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED + index,
    )

    comparisons.append(result)


comparison_df = pd.DataFrame(comparisons)

comparison_df.to_csv(
    "backlog_bootstrap_comparison.csv",
    index=False,
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

summary_df = pd.DataFrame(summaries).set_index("regime_model")

print("\n===== MODEL SUMMARY =====")
print(summary_df.T.to_string())

print("\n===== BOOTSTRAP CONFIDENCE INTERVALS =====")
print(comparison_df.to_string(index=False))

print("\nFiles generated:")
print("- backlog_independent_results.csv")
print("- backlog_markov_results.csv")
print("- backlog_bootstrap_comparison.csv")
