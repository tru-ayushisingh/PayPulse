
from dataclasses import replace

from src.paypulse.simulation import (
    SimulationConfig,
    run_simulation,
    summarize_results,
)


def print_summary(name, summary):
    print(f"\n--- {name} ---")
    print(f"Scenarios: {summary['scenario_count']:,}")
    print(
        "Reserve breach probability: "
        f"{summary['reserve_breach_probability']:.2%}"
    )
    print(
        "Average minimum balance: "
        f"${summary['average_minimum_balance']:,.2f}"
    )
    print(
        "5th percentile ending balance: "
        f"${summary['fifth_percentile_ending_balance']:,.2f}"
    )
    print(
        "Average ending balance: "
        f"${summary['average_ending_balance']:,.2f}"
    )


def main():
    baseline_config = SimulationConfig()

    stress_config = replace(
        baseline_config,
        delay_probability=0.25,
        refund_probability=0.20,
        seed=43,
    )

    _, baseline_results = run_simulation(baseline_config)
    _, stress_results = run_simulation(stress_config)

    print("PayPulse | Synthetic Merchant Cash-Flow Simulator")
    print("All values are illustrative, not real financial data.")

    print_summary(
        "Baseline scenario",
        summarize_results(baseline_config, baseline_results),
    )

    print_summary(
        "Stress scenario",
        summarize_results(stress_config, stress_results),
    )


if __name__ == "__main__":
    main()
