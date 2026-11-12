
import math
import numpy as np

# Same marginal probability in both models
q = 0.08
p_normal = 0.05
p_stress = 0.50

payments_per_day = 10
scenarios = 100000

rng = np.random.default_rng(2026)

# Marginal delay probability
p_average = (1 - q) * p_normal + q * p_stress

# Model A: independent delays
independent = (
    rng.random((scenarios, payments_per_day)) < p_average
)

# Model B: shared disruption state
disrupted = rng.random(scenarios) < q

daily_probability = np.where(
    disrupted,
    p_stress,
    p_normal,
)

correlated = (
    rng.random((scenarios, payments_per_day))
    < daily_probability[:, None]
)

# Count delayed payments per day
independent_counts = independent.sum(axis=1)
correlated_counts = correlated.sum(axis=1)

# Compare probability of 3 or more delays
independent_tail = np.mean(independent_counts >= 3)
correlated_tail = np.mean(correlated_counts >= 3)

# Analytical binomial tail probability
def binomial_tail(n, p, minimum):
    return sum(
        math.comb(n, k) * p**k * (1-p)**(n-k)
        for k in range(minimum, n + 1)
    )

theoretical_independent_tail = binomial_tail(
    payments_per_day,
    p_average,
    3,
)

theoretical_correlated_tail = (
    (1 - q) * binomial_tail(
        payments_per_day, p_normal, 3
    )
    + q * binomial_tail(
        payments_per_day, p_stress, 3
    )
)

print("\n--- PAYMENT CLUSTERING COMPARISON ---")

print(f"Average delay probability: {p_average:.4f}")

print("\nAverage delayed payments per day")
print(f"Independent: {independent_counts.mean():.4f}")
print(f"Correlated:  {correlated_counts.mean():.4f}")

print("\nProbability of 3 or more delays in one day")
print(f"Independent empirical: {independent_tail:.4%}")
print(f"Independent theoretical: {theoretical_independent_tail:.4%}")
print(f"Correlated empirical: {correlated_tail:.4%}")
print(f"Correlated theoretical: {theoretical_correlated_tail:.4%}")

print("\nRisk amplification")
print(
    f"{correlated_tail / independent_tail:.2f}x"
)
