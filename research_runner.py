
from src.paypulse.simulation import SimulationConfig
from src.paypulse.research import run_quantitative_research


def main():

    config = SimulationConfig(
        initial_balance=6000,
        daily_expenses=650,
        payments_per_day=10,
        payment_amount=60,
        reserve_threshold=5600,
        days=7,
        scenarios=5000,
        delay_probability=0.08,
        refund_probability=0.08,
        seed=42,
    )

    research = run_quantitative_research(config)

    print("\n========== PAYPULSE QUANTITATIVE RESEARCH ==========")

    print("\n1. DETERMINISTIC BASELINE")

    print(
        research["deterministic_baseline"].to_string(
            index=False
        )
    )

    print("\n2. MONTE CARLO RISK SUMMARY")

    for metric, value in research["simulation_summary"].items():
        print(f"{metric}: {value}")

    print("\n3. SENSITIVITY ANALYSIS")

    print(
        research["sensitivity_analysis"].to_string(
            index=False
        )
    )

    research["sensitivity_analysis"].to_csv(
        "sensitivity_results.csv",
        index=False,
    )

    print("\nResults exported to sensitivity_results.csv")


if __name__ == "__main__":
    main()
