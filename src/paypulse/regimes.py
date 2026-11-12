
import numpy as np


def _validate_dimensions(scenarios, days):
    if scenarios < 1:
        raise ValueError("Scenarios must be positive.")

    if days < 1:
        raise ValueError("Days must be positive.")


def _validate_probability(probability, name):
    if not 0 <= probability <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def generate_independent_regimes(
    scenarios,
    days,
    disruption_probability=0.08,
    seed=42,
):
    """
    Generate independent daily disruption states.

    Returns a boolean array:
    True  = stress
    False = normal
    """

    _validate_dimensions(scenarios, days)

    _validate_probability(
        disruption_probability,
        "Disruption probability",
    )

    rng = np.random.default_rng(seed)

    return (
        rng.random((scenarios, days))
        < disruption_probability
    )


def generate_markov_regimes(
    scenarios,
    days,
    normal_to_stress=0.04,
    stress_to_normal=0.46,
    initial_stress_probability=None,
    seed=42,
):
    """
    Generate persistent disruption states using
    a two-state Markov chain.

    States:
    False = normal
    True  = stress

    If initial_stress_probability is omitted,
    initialize using the stationary distribution.
    """

    _validate_dimensions(scenarios, days)

    _validate_probability(
        normal_to_stress,
        "Normal-to-stress probability",
    )

    _validate_probability(
        stress_to_normal,
        "Stress-to-normal probability",
    )

    if normal_to_stress + stress_to_normal == 0:
        raise ValueError(
            "At least one transition probability must be positive."
        )

    if initial_stress_probability is None:

        initial_stress_probability = (
            normal_to_stress
            / (normal_to_stress + stress_to_normal)
        )

    _validate_probability(
        initial_stress_probability,
        "Initial stress probability",
    )

    rng = np.random.default_rng(seed)

    states = np.zeros(
        (scenarios, days),
        dtype=bool,
    )

    # Initialize the first day.
    states[:, 0] = (
        rng.random(scenarios)
        < initial_stress_probability
    )

    # Generate subsequent days.
    for day in range(1, days):

        previous_stress = states[:, day - 1]

        transition_probability = np.where(
            previous_stress,
            1 - stress_to_normal,
            normal_to_stress,
        )

        states[:, day] = (
            rng.random(scenarios)
            < transition_probability
        )

    return states
