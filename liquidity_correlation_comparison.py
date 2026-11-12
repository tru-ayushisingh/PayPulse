
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. EXPERIMENT CONFIGURATION
# --------------------------------------------------

INITIAL_BALANCE = 6000.0
DAILY_EXPENSES = 650.0

PAYMENTS_PER_DAY = 10
PAYMENT_AMOUNT = 60.0

RESERVE_THRESHOLD = 5600.0

DAYS = 7
SCENARIOS = 20000

NORMAL_DELAY_PROBABILITY = 0.05
STRESS_DELAY_PROBABILITY = 0.50
DISRUPTION_PROBABILITY = 0.08

SEED = 2026


# --------------------------------------------------
# 2. COMMON RANDOM NUMBERS
# --------------------------------------------------

rng = np.random.default_rng(SEED)

# Reuse the same payment-level random numbers
# for both models to make the comparison fair.
payment_random_numbers = rng.random(
    (SCENARIOS, DAYS, PAYMENTS_PER_DAY)
)

# Shared disruption state for each scenario/day.
disruption_random_numbers = rng.random(
    (SCENARIOS, DAYS)
)

# Expected marginal delay probability.
average_delay_probability = (
    (1 - DISRUPTION_PROBABILITY)
    * NORMAL_DELAY_PROBABILITY
    + DISRUPTION_PROBABILITY
    * STRESS_DELAY_PROBABILITY
)


# --------------------------------------------------
# 3. GENERATE DELAY MATRICES
# --------------------------------------------------

# Model A: Independent delays
independent_delays = (
    payment_random_numbers < average_delay_probability
)

# Model B: Correlated delays
disrupted = (
    disruption_random_numbers < DISRUPTION_PROBABILITY
)

daily_delay_probabilities = np.where(
    disrupted,
    STRESS_DELAY_PROBABILITY,
    NORMAL_DELAY_PROBABILITY,
)

correlated_delays = (
    payment_random_numbers
    < daily_delay_probabilities[:, :, None]
)


# --------------------------------------------------
# 4. CASH FLOW SIMULATION
# --------------------------------------------------

def simulate_balances(delays):

    balances = np.zeros((SCENARIOS, DAYS + 1))
    balances[:, 0] = INITIAL_BALANCE

    pending_captures = np.zeros(SCENARIOS)

    for day in range(DAYS):

        # Payments delayed yesterday arrive today.
        received_from_previous_day = pending_captures

        # Today's payments that are not delayed.
        captured_today = (
            (~delays[:, day]).sum(axis=1)
            * PAYMENT_AMOUNT
        )

        # Today's delayed payments arrive tomorrow.
        pending_captures = (
            delays[:, day].sum(axis=1)
            * PAYMENT_AMOUNT
        )

        total_receipts = (
            received_from_previous_day + captured_today
        )

        net_cash_flow = total_receipts - DAILY_EXPENSES

        balances[:, day + 1] = (
            balances[:, day] + net_cash_flow
        )

    return balances


independent_balances = simulate_balances(
    independent_delays
)

correlated_balances = simulate_balances(
    correlated_delays
)


# --------------------------------------------------
# 5. RISK METRICS
# --------------------------------------------------

def calculate_metrics(balances):

    minimum_balance = balances[:, 1:].min(axis=1)

    ending_balance = balances[:, -1]

    shortfall = np.maximum(
        0,
        RESERVE_THRESHOLD - minimum_balance
    )

    var_95 = np.quantile(shortfall, 0.95)

    cvar_95 = (
        var_95
        + np.maximum(shortfall - var_95, 0).mean() / 0.05
    )

    return {
        "reserve_breach_probability": np.mean(
            minimum_balance < RESERVE_THRESHOLD
        ),
        "average_minimum_balance": np.mean(
            minimum_balance
        ),
        "average_ending_balance": np.mean(
            ending_balance
        ),
        "liquidity_var_95": var_95,
        "liquidity_cvar_95": cvar_95,
        "average_reserve_shortfall": np.mean(
            shortfall
        ),
    }


independent_metrics = calculate_metrics(
    independent_balances
)

correlated_metrics = calculate_metrics(
    correlated_balances
)


# --------------------------------------------------
# 6. DISPLAY COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame(
    {
        "Independent": independent_metrics,
        "Correlated": correlated_metrics,
    }
)

print("\n--- PAYPULSE LIQUIDITY RISK COMPARISON ---")

print(f"Scenarios: {SCENARIOS:,}")
print(f"Simulation horizon: {DAYS} days")
print(f"Expected delay probability: {average_delay_probability:.2%}")

print("\n", comparison.round(4).to_string())

# Difference in reserve-breach probability.
breach_difference = (
    correlated_metrics["reserve_breach_probability"]
    - independent_metrics["reserve_breach_probability"]
)

print("\n--- CORRELATION IMPACT ---")

print(
    f"Absolute breach probability difference: "
    f"{breach_difference:.2%}"
)

if independent_metrics["reserve_breach_probability"] > 0:

    amplification = (
        correlated_metrics["reserve_breach_probability"]
        / independent_metrics["reserve_breach_probability"]
    )

    print(f"Breach probability ratio: {amplification:.2f}x")

print("\nValidation checks")

assert abs(
    independent_delays.mean() - average_delay_probability
) < 0.01

assert abs(
    correlated_delays.mean() - average_delay_probability
) < 0.01

print("Both models have approximately the same marginal delay rate.")
print("Liquidity comparison completed successfully.")



# --------------------------------------------------
# 7. PAIRED STATISTICAL COMPARISON
# --------------------------------------------------

independent_minimum = independent_balances[:, 1:].min(axis=1)
correlated_minimum = correlated_balances[:, 1:].min(axis=1)

independent_breaches = (
    independent_minimum < RESERVE_THRESHOLD
)

correlated_breaches = (
    correlated_minimum < RESERVE_THRESHOLD
)

# Positive means correlation increased the metric.
# Negative means correlation decreased the metric.
breach_differences = (
    correlated_breaches.astype(float)
    - independent_breaches.astype(float)
)

shortfall_independent = np.maximum(
    0,
    RESERVE_THRESHOLD - independent_minimum,
)

shortfall_correlated = np.maximum(
    0,
    RESERVE_THRESHOLD - correlated_minimum,
)

shortfall_differences = (
    shortfall_correlated - shortfall_independent
)


def paired_confidence_interval(differences):

    mean_difference = differences.mean()

    standard_error = (
        differences.std(ddof=1)
        / np.sqrt(len(differences))
    )

    lower = mean_difference - 1.96 * standard_error
    upper = mean_difference + 1.96 * standard_error

    return mean_difference, lower, upper


breach_mean, breach_low, breach_high = (
    paired_confidence_interval(breach_differences)
)

shortfall_mean, shortfall_low, shortfall_high = (
    paired_confidence_interval(shortfall_differences)
)


print("\n--- PAIRED STATISTICAL VALIDATION ---")

print("\nReserve breach probability difference")
print(f"Mean difference: {breach_mean:.4%}")
print(f"95% CI: [{breach_low:.4%}, {breach_high:.4%}]")

print("\nAverage reserve shortfall difference")
print(f"Mean difference: {shortfall_mean:.2f}")
print(f"95% CI: [{shortfall_low:.2f}, {shortfall_high:.2f}]")


print("\n--- SHORTFALL DISTRIBUTION ---")

for percentile in [50, 90, 95, 99, 99.5]:

    independent_value = np.percentile(
        shortfall_independent,
        percentile,
    )

    correlated_value = np.percentile(
        shortfall_correlated,
        percentile,
    )

    print(
        f"P{percentile}: "
        f"Independent={independent_value:.2f}, "
        f"Correlated={correlated_value:.2f}"
    )



# --------------------------------------------------
# 8. RESERVE THRESHOLD SENSITIVITY
# --------------------------------------------------

thresholds = [
    5400,
    5450,
    5500,
    5550,
    5600,
    5650,
    5700,
    5750,
    5800,
    5850,
    5900,
]

sensitivity_rows = []

independent_minimum = independent_balances[:, 1:].min(axis=1)
correlated_minimum = correlated_balances[:, 1:].min(axis=1)

for threshold in thresholds:

    independent_breach = (
        independent_minimum < threshold
    )

    correlated_breach = (
        correlated_minimum < threshold
    )

    independent_probability = independent_breach.mean()
    correlated_probability = correlated_breach.mean()

    sensitivity_rows.append({
        "reserve_threshold": threshold,
        "independent_breach_probability": independent_probability,
        "correlated_breach_probability": correlated_probability,
        "difference_percentage_points": (
            correlated_probability - independent_probability
        ) * 100,
    })


sensitivity_df = pd.DataFrame(sensitivity_rows)

print("\n--- RESERVE THRESHOLD SENSITIVITY ---")
print(sensitivity_df.round(4).to_string(index=False))

sensitivity_df.to_csv(
    "reserve_threshold_sensitivity.csv",
    index=False,
)

print("\nSaved: reserve_threshold_sensitivity.csv")
