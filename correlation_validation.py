
import numpy as np

from src.paypulse.event_simulation import (
    EventSimulationConfig,
    run_event_simulation,
)


config = EventSimulationConfig(
    initial_balance=1000,
    daily_expenses=0,
    payments_per_day=10,
    payment_amount=1,
    reserve_threshold=0,
    days=1,
    scenarios=10000,

    normal_delay_probability=0.05,
    stress_delay_probability=0.50,
    disruption_probability=0.08,

    refund_probability=0,
    seed=2026,
)

daily, results, events = run_event_simulation(config)


# Keep only newly generated payment outcomes.
payment_events = events[
    events["event_type"].isin([
        "payment_delayed",
        "payment_captured",
    ])
].copy()

payment_events["is_delayed"] = (
    payment_events["event_type"] == "payment_delayed"
).astype(int)


# Count delayed payments in each scenario/day cluster.
delayed_counts = (
    payment_events
    .groupby(["scenario", "day"])["is_delayed"]
    .sum()
    .to_numpy()
)

payments = config.payments_per_day
clusters = len(delayed_counts)

# Empirical marginal delay probability.
total_payments = clusters * payments

empirical_delay_rate = (
    delayed_counts.sum() / total_payments
)

# Probability that two distinct payments in the
# same scenario/day are both delayed.
empirical_joint_probability = (
    np.sum(delayed_counts * (delayed_counts - 1))
    / (clusters * payments * (payments - 1))
)

# Empirical pairwise correlation.
empirical_correlation = (
    empirical_joint_probability - empirical_delay_rate**2
) / (
    empirical_delay_rate * (1 - empirical_delay_rate)
)


# Theoretical values.
q = config.disruption_probability
p_normal = config.normal_delay_probability
p_stress = config.stress_delay_probability

theoretical_delay_rate = (
    (1 - q) * p_normal + q * p_stress
)

theoretical_joint_probability = (
    (1 - q) * p_normal**2 + q * p_stress**2
)

theoretical_correlation = (
    theoretical_joint_probability - theoretical_delay_rate**2
) / (
    theoretical_delay_rate * (1 - theoretical_delay_rate)
)


print("\n--- PAYPULSE CORRELATION VALIDATION ---")

print(f"Scenarios: {config.scenarios}")
print(f"Payments per scenario/day: {payments}")

print("\nMarginal delay probability")
print(f"Empirical:   {empirical_delay_rate:.4f}")
print(f"Theoretical: {theoretical_delay_rate:.4f}")

print("\nPairwise delay correlation")
print(f"Empirical:   {empirical_correlation:.4f}")
print(f"Theoretical: {theoretical_correlation:.4f}")

print("\nDifference")
print(
    f"{abs(empirical_correlation - theoretical_correlation):.4f}"
)

assert abs(
    empirical_correlation - theoretical_correlation
) < 0.03, "Empirical correlation differs too much from theory."

print("\nValidation passed!")
