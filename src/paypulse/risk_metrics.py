
import numpy as np


def calculate_shortfall(minimum_balances, reserve_threshold):
    """
    Calculate reserve shortfall for each simulation.
    """

    minimum_balances = np.asarray(
        minimum_balances,
        dtype=float,
    )

    if minimum_balances.size == 0:
        raise ValueError("Minimum balances cannot be empty.")

    if not np.all(np.isfinite(minimum_balances)):
        raise ValueError("Minimum balances must be finite.")

    if not np.isfinite(reserve_threshold):
        raise ValueError("Reserve threshold must be finite.")

    return np.maximum(
        0,
        reserve_threshold - minimum_balances,
    )


def calculate_var(losses, confidence_level=0.95):
    """
    Calculate empirical Value at Risk.
    """

    losses = np.asarray(losses, dtype=float)

    if losses.size == 0:
        raise ValueError("Losses cannot be empty.")

    if not np.all(np.isfinite(losses)):
        raise ValueError("Losses must be finite.")

    if np.any(losses < 0):
        raise ValueError("Losses cannot be negative.")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    return float(
        np.quantile(losses, confidence_level)
    )


def calculate_cvar(losses, confidence_level=0.95):
    """
    Calculate empirical CVaR using the
    Rockafellar-Uryasev formulation.
    """

    losses = np.asarray(losses, dtype=float)

    var = calculate_var(
        losses,
        confidence_level,
    )

    alpha = confidence_level

    return float(
        var
        + np.maximum(losses - var, 0).mean()
        / (1 - alpha)
    )


def calculate_risk_metrics(
    minimum_balances,
    ending_balances,
    reserve_threshold,
    confidence_level=0.95,
):
    """
    Calculate a consolidated set of liquidity risk metrics.
    """

    minimum_balances = np.asarray(
        minimum_balances,
        dtype=float,
    )

    ending_balances = np.asarray(
        ending_balances,
        dtype=float,
    )

    if minimum_balances.size == 0:
        raise ValueError("Minimum balances cannot be empty.")

    if minimum_balances.shape != ending_balances.shape:
        raise ValueError("Balance arrays must have matching shapes.")

    if not np.all(np.isfinite(ending_balances)):
        raise ValueError("Ending balances must be finite.")

    shortfalls = calculate_shortfall(
        minimum_balances,
        reserve_threshold,
    )

    return {
        "scenario_count": len(minimum_balances),

        "reserve_breach_probability": float(
            np.mean(minimum_balances < reserve_threshold)
        ),

        "average_minimum_balance": float(
            minimum_balances.mean()
        ),

        "average_ending_balance": float(
            ending_balances.mean()
        ),

        "liquidity_var_95": calculate_var(
            shortfalls,
            confidence_level,
        ),

        "liquidity_cvar_95": calculate_cvar(
            shortfalls,
            confidence_level,
        ),

        "average_reserve_shortfall": float(
            shortfalls.mean()
        ),
    }
