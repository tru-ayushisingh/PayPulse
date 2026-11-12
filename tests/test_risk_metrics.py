
import numpy as np
import pytest

from src.paypulse.risk_metrics import (
    calculate_shortfall,
    calculate_var,
    calculate_cvar,
    calculate_risk_metrics,
)


def test_shortfall_calculation():
    minimum = np.array([5000, 5500, 5700])

    result = calculate_shortfall(minimum, 5600)

    assert np.array_equal(
        result,
        np.array([600, 100, 0]),
    )


def test_var_calculation():
    losses = np.array([0, 10, 20, 30, 40])

    result = calculate_var(losses, 0.80)

    assert result == 32.0


def test_cvar_not_less_than_var():
    losses = np.array([0, 10, 20, 30, 100])

    var = calculate_var(losses)
    cvar = calculate_cvar(losses)

    assert cvar >= var


def test_risk_metrics_breach_probability():
    minimum = np.array([5000, 5500, 5700, 5800])
    ending = np.array([5200, 5600, 5750, 5900])

    result = calculate_risk_metrics(
        minimum,
        ending,
        reserve_threshold=5600,
    )

    assert result["reserve_breach_probability"] == 0.5


def test_empty_losses_rejected():
    with pytest.raises(ValueError):
        calculate_var(np.array([]))


def test_negative_losses_rejected():
    with pytest.raises(ValueError):
        calculate_var(np.array([-10, 20]))


def test_invalid_confidence_level_rejected():
    with pytest.raises(ValueError):
        calculate_var(
            np.array([10, 20]),
            confidence_level=1.0,
        )


def test_mismatched_balance_shapes_rejected():
    with pytest.raises(ValueError):
        calculate_risk_metrics(
            np.array([5000, 5500]),
            np.array([5100]),
            reserve_threshold=5600,
        )
