
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

NORMAL_DELAY_PROBABILITY = 0.05
STRESS_DELAY_PROBABILITY = 0.50

NORMAL_RESOLUTION_PROBABILITY = 0.75
STRESS_RESOLUTION_PROBABILITY = 0.25

DISRUPTION_PROBABILITY = 0.08

NORMAL_TO_STRESS = 0.04
STRESS_TO_NORMAL = 0.46

SEED = 2027


# --------------------------------------------------
# 2. GENERATE DISRUPTION REGIMES
# --------------------------------------------------

def generate_independent_regimes(rng):

    return (
        rng.random((SCENARIOS, DAYS))
        < DISRUPTION_PROBABILITY
    )


def generate_markov_regimes(rng):

    states = np.zeros(
        (SCENARIOS, DAYS),
        dtype=bool,
    )

    # Start from stationary distribution.
    states[:, 0] = (
        rng.random(SCENARIOS)
        < DISRUPTION_PROBABILITY
    )

    for day in range(1, DAYS):

        previous_stress = states[:, day - 1]

        transition_probability = np.where(
            previous_stress,
            1 - STRESS_TO_NORMAL,
            NORMAL_TO_STRESS,
        )

        states[:, day] = (
            rng.random(SCENARIOS)
            < transition_probability
        )

    return states


# --------------------------------------------------
# 3. MULTI-DAY PAYMENT SIMULATION
# --------------------------------------------------

def simulate_payment_backlog(regimes, seed):

    rng = np.random.default_rng(seed)

    balances = np.zeros(
        (SCENARIOS, DAYS + 1)
    )

    balances[:, 0] = INITIAL_BALANCE

    # Number of payments still awaiting capture.
    pending_payments = np.zeros(
        SCENARIOS,
        dtype=int,
    )

    backlog_history = np.zeros(
        (SCENARIOS, DAYS)
    )

    for day in range(DAYS):

        disrupted = regimes[:, day]

        # Today's regime determines the probability
        # of resolving previously pending payments.
        resolution_probability = np.where(
            disrupted,
            STRESS_RESOLUTION_PROBABILITY,
            NORMAL_RESOLUTION_PROBABILITY,
        )

        # Resolve some previously pending payments.
        resolved_payments = rng.binomial(
            pending_payments,
            resolution_probability,
        )

        pending_payments -= resolved_payments

        # Today's regime also affects new payment delays.
        delay_probability = np.where(
            disrupted,
            STRESS_DELAY_PROBABILITY,
            NORMAL_DELAY_PROBABILITY,
        )

        delayed_today = rng.binomial(
            PAYMENTS_PER_DAY,
            delay_probability,
        )

        captured_today = (
            PAYMENTS_PER_DAY - delayed_today
        )

        # Add newly delayed payments to the backlog.
        pending_payments += delayed_today

        # Previously pending payments resolved today
        # plus new payments captured today.
        total_captured = (
            resolved_payments + captured_today
        )

        receipts = total_captured * PAYMENT_AMOUNT

        net_cash_flow = receipts - DAILY_EXPENSES

        balances[:, day + 1] = (
            balances[:, day] + net_cash_flow
        )

        backlog_history[:, day] = pending_payments

    return balances, backlog_history


# --------------------------------------------------
# 4. RISK METRICS
# --------------------------------------------------

def calculate_metrics(balances, backlog_history):

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
        "reserve_breach_probability": np.mean(
            minimum_balance < RESERVE_THRESHOLD
        ),

        "average_minimum_balance": np.mean(
            minimum_balance
        ),

        "average_ending_balance": np.mean(
            balances[:, -1]
        ),

        "average_reserve_shortfall": np.mean(
            shortfall
        ),

        "liquidity_var_95": var_95,

        "liquidity_cvar_95": cvar_95,

        "average_outstanding_payments": np.mean(
            backlog_history
        ),

        "average_final_backlog": np.mean(
            backlog_history[:, -1]
        ),

        "average_peak_backlog": np.mean(
            backlog_history.max(axis=1)
        ),
    }


# --------------------------------------------------
# 5. RUN BOTH EXPERIMENTS
# --------------------------------------------------

regime_rng = np.random.default_rng(SEED)

independent_regimes = generate_independent_regimes(
    regime_rng
)

markov_regimes = generate_markov_regimes(
    regime_rng
)

independent_balances, independent_backlog = (
    simulate_payment_backlog(
        independent_regimes,
        seed=SEED + 1,
    )
)

markov_balances, markov_backlog = (
    simulate_payment_backlog(
        markov_regimes,
        seed=SEED + 2,
    )
)


# --------------------------------------------------
# 6. COMPARE RESULTS
# --------------------------------------------------

independent_metrics = calculate_metrics(
    independent_balances,
    independent_backlog,
)

markov_metrics = calculate_metrics(
    markov_balances,
    markov_backlog,
)

comparison = pd.DataFrame({
    "Independent regime": independent_metrics,
    "Persistent Markov regime": markov_metrics,
})

print("\n--- MULTI-DAY PAYMENT DELAY EXPERIMENT ---")

print(f"Scenarios: {SCENARIOS:,}")
print(f"Days: {DAYS}")
print(f"Reserve threshold: {RESERVE_THRESHOLD}")

print("\nRisk and backlog comparison")
print(comparison.round(4).to_string())


# --------------------------------------------------
# 7. MODEL DIAGNOSTICS
# --------------------------------------------------

print("\n--- REGIME DIAGNOSTICS ---")

print(
    "Independent stress frequency:",
    round(independent_regimes.mean(), 4),
)

print(
    "Markov stress frequency:",
    round(markov_regimes.mean(), 4),
)

print("\n--- PAYMENT DELAY DIAGNOSTICS ---")

print(
    "Independent final backlog:",
    round(independent_backlog[:, -1].mean(), 4),
)

print(
    "Markov final backlog:",
    round(markov_backlog[:, -1].mean(), 4),
)


# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

comparison.to_csv(
    "multi_day_liquidity_comparison.csv"
)

print("\nSaved: multi_day_liquidity_comparison.csv")
print("\nExperiment completed.")


# --------------------------------------------------
# 9. STATISTICAL VALIDATION
# --------------------------------------------------

from math import sqrt


def get_scenario_shortfalls(balances):

    minimum_balance = balances[:, 1:].min(axis=1)

    return np.maximum(
        0,
        RESERVE_THRESHOLD - minimum_balance,
    )


independent_shortfalls = get_scenario_shortfalls(
    independent_balances
)

markov_shortfalls = get_scenario_shortfalls(
    markov_balances
)


def difference_confidence_interval(a, b):

    difference = b.mean() - a.mean()

    se = sqrt(
        a.var(ddof=1) / len(a)
        + b.var(ddof=1) / len(b)
    )

    return {
        "difference": difference,
        "lower": difference - 1.96 * se,
        "upper": difference + 1.96 * se,
    }


# Reserve breach probability
independent_breaches = (
    independent_shortfalls > 0
).astype(float)

markov_breaches = (
    markov_shortfalls > 0
).astype(float)

breach_ci = difference_confidence_interval(
    independent_breaches,
    markov_breaches,
)

# Average reserve shortfall
shortfall_ci = difference_confidence_interval(
    independent_shortfalls,
    markov_shortfalls,
)


# Bootstrap CVaR confidence intervals.
def calculate_cvar(losses):

    var_95 = np.quantile(losses, 0.95)

    return (
        var_95
        + np.maximum(losses - var_95, 0).mean() / 0.05
    )


def bootstrap_cvar_interval(losses, seed, repetitions=500):

    rng = np.random.default_rng(seed)

    estimates = np.empty(repetitions)

    for i in range(repetitions):

        sample = rng.choice(
            losses,
            size=len(losses),
            replace=True,
        )

        estimates[i] = calculate_cvar(sample)

    return np.quantile(
        estimates,
        [0.025, 0.975],
    )


independent_cvar_ci = bootstrap_cvar_interval(
    independent_shortfalls,
    SEED + 10,
)

markov_cvar_ci = bootstrap_cvar_interval(
    markov_shortfalls,
    SEED + 20,
)


print("\n--- MULTI-DAY STATISTICAL VALIDATION ---")

print("\nReserve breach probability difference")
print(f"Difference: {breach_ci['difference']:.4%}")
print(
    f"95% CI: "
    f"[{breach_ci['lower']:.4%}, "
    f"{breach_ci['upper']:.4%}]"
)

print("\nAverage shortfall difference")
print(f"Difference: {shortfall_ci['difference']:.2f}")
print(
    f"95% CI: "
    f"[{shortfall_ci['lower']:.2f}, "
    f"{shortfall_ci['upper']:.2f}]"
)

print("\nCVaR 95 bootstrap confidence intervals")
print(
    "Independent:",
    np.round(independent_cvar_ci, 2),
)

print(
    "Persistent:",
    np.round(markov_cvar_ci, 2),
)



# --------------------------------------------------
# 10. BOOTSTRAP CONFIDENCE INTERVAL FOR CVaR DIFFERENCE
# --------------------------------------------------

def bootstrap_cvar_difference(
    independent_losses,
    persistent_losses,
    repetitions=2000,
    seed=2030,
):

    rng = np.random.default_rng(seed)

    differences = np.empty(repetitions)

    for i in range(repetitions):

        independent_sample = rng.choice(
            independent_losses,
            size=len(independent_losses),
            replace=True,
        )

        persistent_sample = rng.choice(
            persistent_losses,
            size=len(persistent_losses),
            replace=True,
        )

        independent_cvar = calculate_cvar(
            independent_sample
        )

        persistent_cvar = calculate_cvar(
            persistent_sample
        )

        differences[i] = (
            persistent_cvar - independent_cvar
        )

    return {
        "mean_difference": differences.mean(),
        "lower_95": np.quantile(differences, 0.025),
        "upper_95": np.quantile(differences, 0.975),
    }


cvar_difference = bootstrap_cvar_difference(
    independent_shortfalls,
    markov_shortfalls,
)

print("\n--- CVaR DIFFERENCE VALIDATION ---")

print(
    "Persistent minus independent CVaR:",
    round(cvar_difference["mean_difference"], 2),
)

print(
    "95% bootstrap CI:",
    round(cvar_difference["lower_95"], 2),
    "to",
    round(cvar_difference["upper_95"], 2),
)
