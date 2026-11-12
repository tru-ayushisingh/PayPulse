
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

INITIAL_BALANCE = 6000.0
DAILY_EXPENSES = 650.0

PAYMENTS_PER_DAY = 10
PAYMENT_AMOUNT = 60.0

RESERVE_THRESHOLD = 5500.0

DAYS = 7
SCENARIOS = 20000

NORMAL_DELAY = 0.05
STRESS_DELAY = 0.50

DISRUPTION_PROBABILITY = 0.08

# Markov transition probabilities
NORMAL_TO_STRESS = 0.04
STRESS_TO_NORMAL = 0.46

SEED = 2026

rng = np.random.default_rng(SEED)


# --------------------------------------------------
# 2. COMMON PAYMENT RANDOM NUMBERS
# --------------------------------------------------

payment_random_numbers = rng.random(
    (SCENARIOS, DAYS, PAYMENTS_PER_DAY)
)

# Independent daily disruption states
iid_disruptions = (
    rng.random((SCENARIOS, DAYS))
    < DISRUPTION_PROBABILITY
)


# --------------------------------------------------
# 3. GENERATE PERSISTENT MARKOV REGIMES
# --------------------------------------------------

markov_disruptions = np.zeros(
    (SCENARIOS, DAYS),
    dtype=bool,
)

# Start from the stationary distribution.
markov_disruptions[:, 0] = (
    rng.random(SCENARIOS)
    < DISRUPTION_PROBABILITY
)

for day in range(1, DAYS):

    previous_stress = markov_disruptions[:, day - 1]

    transition_probability = np.where(
        previous_stress,
        1 - STRESS_TO_NORMAL,
        NORMAL_TO_STRESS,
    )

    markov_disruptions[:, day] = (
        rng.random(SCENARIOS)
        < transition_probability
    )


# --------------------------------------------------
# 4. GENERATE PAYMENT DELAYS
# --------------------------------------------------

iid_delay_probabilities = np.where(
    iid_disruptions,
    STRESS_DELAY,
    NORMAL_DELAY,
)

markov_delay_probabilities = np.where(
    markov_disruptions,
    STRESS_DELAY,
    NORMAL_DELAY,
)

iid_delays = (
    payment_random_numbers
    < iid_delay_probabilities[:, :, None]
)

markov_delays = (
    payment_random_numbers
    < markov_delay_probabilities[:, :, None]
)


# --------------------------------------------------
# 5. CASH BALANCE SIMULATION
# --------------------------------------------------

def simulate_balances(delays):

    balances = np.zeros(
        (SCENARIOS, DAYS + 1)
    )

    balances[:, 0] = INITIAL_BALANCE

    pending_captures = np.zeros(SCENARIOS)

    for day in range(DAYS):

        received_from_previous_day = pending_captures

        captured_today = (
            (~delays[:, day]).sum(axis=1)
            * PAYMENT_AMOUNT
        )

        pending_captures = (
            delays[:, day].sum(axis=1)
            * PAYMENT_AMOUNT
        )

        receipts = (
            received_from_previous_day
            + captured_today
        )

        balances[:, day + 1] = (
            balances[:, day]
            + receipts
            - DAILY_EXPENSES
        )

    return balances


iid_balances = simulate_balances(iid_delays)

markov_balances = simulate_balances(markov_delays)


# --------------------------------------------------
# 6. RISK METRICS
# --------------------------------------------------

def calculate_metrics(balances):

    minimum_balance = balances[:, 1:].min(axis=1)

    shortfall = np.maximum(
        0,
        RESERVE_THRESHOLD - minimum_balance,
    )

    var_95 = np.quantile(shortfall, 0.95)

    cvar_95 = (
        var_95
        + np.maximum(shortfall - var_95, 0).mean() / 0.05
    )

    return {
        "breach_probability": np.mean(
            minimum_balance < RESERVE_THRESHOLD
        ),
        "average_minimum_balance": minimum_balance.mean(),
        "average_ending_balance": balances[:, -1].mean(),
        "average_shortfall": shortfall.mean(),
        "liquidity_var_95": var_95,
        "liquidity_cvar_95": cvar_95,
    }


iid_metrics = calculate_metrics(iid_balances)

markov_metrics = calculate_metrics(markov_balances)


# --------------------------------------------------
# 7. DISRUPTION PERSISTENCE METRICS
# --------------------------------------------------

def average_longest_stress_run(states):

    longest_runs = []

    for scenario_states in states:

        current_run = 0
        longest_run = 0

        for is_stress in scenario_states:

            if is_stress:
                current_run += 1
                longest_run = max(
                    longest_run,
                    current_run,
                )
            else:
                current_run = 0

        longest_runs.append(longest_run)

    return np.mean(longest_runs)


# --------------------------------------------------
# 8. RESULTS
# --------------------------------------------------

comparison = pd.DataFrame({
    "Independent daily regime": iid_metrics,
    "Persistent Markov regime": markov_metrics,
})

print("\n--- PERSISTENT DISRUPTION EXPERIMENT ---")

print(f"Scenarios: {SCENARIOS:,}")
print(f"Days: {DAYS}")
print(f"Reserve threshold: {RESERVE_THRESHOLD}")

print("\nRisk comparison")
print(comparison.round(4).to_string())

print("\nDisruption diagnostics")

print(
    "Independent stress-day frequency:",
    round(iid_disruptions.mean(), 4),
)

print(
    "Markov stress-day frequency:",
    round(markov_disruptions.mean(), 4),
)

print(
    "Average longest independent stress run:",
    round(
        average_longest_stress_run(iid_disruptions),
        4,
    ),
)

print(
    "Average longest Markov stress run:",
    round(
        average_longest_stress_run(markov_disruptions),
        4,
    ),
)

print("\nAverage payment delay rates")

print(
    "Independent:",
    round(iid_delays.mean(), 4),
)

print(
    "Persistent:",
    round(markov_delays.mean(), 4),
)

print("\nExperiment completed.")


# --------------------------------------------------
# 9. MARKOV PERSISTENCE VALIDATION
# --------------------------------------------------

def calculate_transition_metrics(states):

    previous = states[:, :-1].flatten()
    current = states[:, 1:].flatten()

    normal_to_stress = np.mean(
        current[~previous]
    )

    stress_to_stress = np.mean(
        current[previous]
    )

    stress_to_normal = 1 - stress_to_stress

    return {
        "normal_to_stress": normal_to_stress,
        "stress_to_stress": stress_to_stress,
        "stress_to_normal": stress_to_normal,
        "stress_frequency": states.mean(),
    }


iid_transition_metrics = calculate_transition_metrics(
    iid_disruptions
)

markov_transition_metrics = calculate_transition_metrics(
    markov_disruptions
)

print("\n--- MARKOV PERSISTENCE VALIDATION ---")

print("\nExpected transition probabilities")
print("Normal -> Stress:", NORMAL_TO_STRESS)
print("Stress -> Stress:", 1 - STRESS_TO_NORMAL)

print("\nIndependent regime")
for key, value in iid_transition_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nPersistent Markov regime")
for key, value in markov_transition_metrics.items():
    print(f"{key}: {value:.4f}")
