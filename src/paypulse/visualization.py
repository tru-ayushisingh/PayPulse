
import matplotlib.pyplot as plt

from src.paypulse.research import run_quantitative_research
from src.paypulse.simulation import SimulationConfig


def create_research_charts(config: SimulationConfig):

    research = run_quantitative_research(config)

    baseline = research["deterministic_baseline"]
    daily = research["daily_scenarios"]
    sensitivity = research["sensitivity_analysis"]

    # Chart 1: Cash-flow uncertainty.
    days = range(config.days + 1)

    p05 = daily.quantile(0.05, axis=0).to_numpy()
    p50 = daily.quantile(0.50, axis=0).to_numpy()
    p95 = daily.quantile(0.95, axis=0).to_numpy()

    plt.figure(figsize=(10, 5))

    plt.fill_between(
        days,
        p05,
        p95,
        alpha=0.2,
        label="5th–95th percentile range",
    )

    plt.plot(days, p50, label="Median simulated balance")
    plt.plot(
        baseline["day"],
        baseline["expected_balance"],
        linestyle="--",
        label="Deterministic expected balance",
    )

    plt.axhline(
        config.reserve_threshold,
        linestyle=":",
        label="Minimum reserve",
    )

    plt.xlabel("Day")
    plt.ylabel("Cash balance ($)")
    plt.title("PayPulse: Cash-Flow Uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cashflow_uncertainty.png")
    plt.close()

    # Chart 2: Sensitivity analysis.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    parameters = [
        "delay_probability",
        "refund_probability",
        "initial_balance",
    ]

    for ax, parameter in zip(axes, parameters):

        subset = sensitivity[
            sensitivity["parameter"] == parameter
        ]

        ax.plot(
            subset["value"],
            subset["breach_probability"],
            marker="o",
        )

        ax.set_title(parameter.replace("_", " ").title())
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Reserve breach probability")
        ax.grid(True)

    fig.suptitle("PayPulse: Sensitivity Analysis")
    fig.tight_layout()
    fig.savefig("sensitivity_analysis.png")
    plt.close(fig)

    print("Charts generated successfully.")


if __name__ == "__main__":

    config = SimulationConfig()

    create_research_charts(config)
