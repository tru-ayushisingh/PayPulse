
import numpy as np
import pytest

from src.paypulse.regimes import (
    generate_independent_regimes,
    generate_markov_regimes,
)


def test_independent_regimes_dimensions():
    states = generate_independent_regimes(
        scenarios=100,
        days=7,
        seed=42,
    )

    assert states.shape == (100, 7)
    assert states.dtype == bool


def test_independent_regimes_reproducibility():
    first = generate_independent_regimes(
        scenarios=100,
        days=7,
        seed=42,
    )

    second = generate_independent_regimes(
        scenarios=100,
        days=7,
        seed=42,
    )

    assert np.array_equal(first, second)


def test_independent_regimes_probability():
    states = generate_independent_regimes(
        scenarios=20000,
        days=7,
        disruption_probability=0.08,
        seed=42,
    )

    assert abs(states.mean() - 0.08) < 0.01


def test_markov_regimes_dimensions():
    states = generate_markov_regimes(
        scenarios=100,
        days=7,
        seed=42,
    )

    assert states.shape == (100, 7)
    assert states.dtype == bool


def test_markov_regimes_reproducibility():
    first = generate_markov_regimes(
        scenarios=100,
        days=7,
        seed=42,
    )

    second = generate_markov_regimes(
        scenarios=100,
        days=7,
        seed=42,
    )

    assert np.array_equal(first, second)


def test_markov_stationary_probability():
    states = generate_markov_regimes(
        scenarios=20000,
        days=100,
        normal_to_stress=0.04,
        stress_to_normal=0.46,
        seed=42,
    )

    # Ignore the first day to reduce initialization effects.
    observed_frequency = states[:, 1:].mean()

    assert abs(observed_frequency - 0.08) < 0.01


def test_markov_persistence():
    states = generate_markov_regimes(
        scenarios=20000,
        days=100,
        normal_to_stress=0.04,
        stress_to_normal=0.46,
        seed=42,
    )

    previous = states[:, :-1].flatten()
    current = states[:, 1:].flatten()

    observed_stress_persistence = current[previous].mean()

    assert abs(observed_stress_persistence - 0.54) < 0.02


def test_invalid_scenario_count():
    with pytest.raises(ValueError):
        generate_independent_regimes(
            scenarios=0,
            days=7,
        )


def test_invalid_transition_probability():
    with pytest.raises(ValueError):
        generate_markov_regimes(
            scenarios=100,
            days=7,
            normal_to_stress=1.5,
        )


def test_zero_transition_probabilities():
    with pytest.raises(ValueError):
        generate_markov_regimes(
            scenarios=100,
            days=7,
            normal_to_stress=0,
            stress_to_normal=0,
        )
