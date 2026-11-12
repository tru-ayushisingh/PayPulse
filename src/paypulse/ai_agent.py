
import json
import re
from dataclasses import replace

from src.paypulse.ai_client import ask_gemini
from src.paypulse.backlog_simulation import (
    BacklogSimulationConfig,
    run_backlog_simulation,
    summarize_backlog_results,
    validate_backlog_config,
)


ALLOWED_PARAMETERS = {
    "stress_delay_probability",
    "refund_probability",
    "stress_resolution_probability",
    "normal_resolution_probability",
    "regime_model",
    "disruption_probability",
    "normal_to_stress",
    "stress_to_normal",
    "days",
    "payments_per_day",
    "payment_amount",
    "initial_balance",
    "daily_expenses",
    "reserve_threshold",
}


def extract_simulation_parameters(question: str) -> dict:
    """Ask Gemini to convert a merchant question into parameters."""

    prompt = f"""
You are the parameter extraction component of PayPulse.

Extract only simulation parameters explicitly requested or clearly
implied by the merchant's question.

Return ONLY a valid JSON object. Do not include Markdown or explanations.

Allowed keys and meanings:
- stress_delay_probability: probability that a new payment is delayed
- refund_probability: probability that a captured payment is refunded
- stress_resolution_probability: probability that an outstanding payment
  is resolved on a day when the system is stressed
- normal_resolution_probability: probability of resolution in normal conditions
- regime_model: either "independent" or "markov"
- disruption_probability: daily disruption probability for independent regime
- normal_to_stress: Markov probability of moving from normal to stress
- stress_to_normal: Markov probability of moving from stress to normal
- days: simulation horizon in days
- payments_per_day: number of incoming payments per day
- payment_amount: amount of each payment
- initial_balance: starting merchant balance
- daily_expenses: merchant expenses per day
- reserve_threshold: minimum desired balance

Rules:
- Convert percentages to decimals. 20 percent becomes 0.20.
- "Persistent disruption" means regime_model="markov".
- "Independent disruption" means regime_model="independent".
- Do not invent values that the merchant did not specify.
- Return an empty object {{}} if no parameters can be extracted.
- Do not return any keys outside the allowed list.
- Do not return explanations.

Merchant question:
{question}
"""

    raw_response = ask_gemini(prompt)

    try:
        parameters = json.loads(raw_response)
    except json.JSONDecodeError:
        # Handle a response wrapped in a Markdown code fence.
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw_response.strip(),
            flags=re.IGNORECASE,
        )
        try:
            parameters = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Gemini did not return valid JSON parameters."
            ) from error

    if not isinstance(parameters, dict):
        raise ValueError("Gemini parameters must be a JSON object.")

    unknown_keys = set(parameters) - ALLOWED_PARAMETERS
    if unknown_keys:
        raise ValueError(
            f"Gemini returned unsupported parameters: {sorted(unknown_keys)}"
        )

    return parameters


def run_ai_analysis(
    question: str,
    base_config: BacklogSimulationConfig | None = None,
) -> dict:
    """Extract parameters, run the simulation, and explain its results."""

    if not question or not question.strip():
        raise ValueError("Please provide a merchant question.")

    if len(question) > 2000:
        raise ValueError("Question must be 2000 characters or fewer.")

    if base_config is None:
        base_config = BacklogSimulationConfig(
            scenarios=1000,
            days=7,
        )

    parameters = extract_simulation_parameters(question)

    # Map the merchant-facing stress delay parameter to the
    # existing event-driven simulation configuration.
    if "stress_delay_probability" in parameters:
        parameters["stress_delay_probability"] = float(
            parameters["stress_delay_probability"]
        )

    integer_fields = {
    "days",
    "payments_per_day",
    }

    for key, value in parameters.items():
        if key in integer_fields:
            if isinstance(value, bool) or int(value) != float(value):
                raise ValueError(f"{key} must be an integer.")
        
            parameters[key] = int(value)

        elif key != "regime_model":
            if isinstance(value, bool):
                raise ValueError(f"{key} must be numeric.")
        parameters[key] = float(value)

    config = replace(base_config, **parameters)

    # Ensure integer configuration fields are actual integers.
    for field in ("days", "payments_per_day", "scenarios"):
        value = getattr(config, field)

        if isinstance(value, bool) or int(value) != float(value):
            raise ValueError(f"{field} must be a whole number.")

        config = replace(config, **{field: int(value)})

    # Validate all AI-generated values before running any simulation.
    validate_backlog_config(config)

    print("DEBUG days:", config.days, type(config.days))
    print("DEBUG scenarios:", config.scenarios, type(config.scenarios))
    print("DEBUG payments_per_day:", config.payments_per_day, type(config.payments_per_day))

    daily, results, events = run_backlog_simulation(config)
    print("DAILY DATA TYPE:", type(daily))
    print("DAILY DATA SAMPLE:", repr(daily)[:2000])
    summary = summarize_backlog_results(config, results)

    # Aggregate all Monte Carlo scenarios into a daily forecast.
    daily_forecast = (
        daily.groupby("day")["balance"]
        .agg(
        average_balance="mean",
        median_balance="median",
        lower_bound=lambda values: values.quantile(0.10),
        upper_bound=lambda values: values.quantile(0.90),
    )
    .reset_index()
    )

# Convert Pandas/NumPy values into standard Python types for JSON.
    daily_forecast = [
        {
        "day": int(row["day"]),
        "average_balance": round(float(row["average_balance"]), 2),
        "median_balance": round(float(row["median_balance"]), 2),
        "lower_bound": round(float(row["lower_bound"]), 2),
        "upper_bound": round(float(row["upper_bound"]), 2),
        }
        for row in daily_forecast.to_dict(orient="records")
    ]

    explanation_prompt = f"""
You are PayPulse AI, a merchant liquidity risk analyst.

Explain the following simulation results to a small business merchant.

Merchant question:
{question}

Parameters applied:
{json.dumps(parameters)}

Simulation results:
{json.dumps(summary)}

f"Configured reserve threshold: ${config.reserve_threshold:.2f}\n"

Important rules:
- Use only the supplied simulation results for numerical claims.
- Do not invent figures, trends, or events.
- Explain reserve breach probability, average shortfall and CVaR
  in understandable business language.
- Explain that the results are simulated estimates, not guarantees.
- Do not describe simulated captures as actual PayPal settlements.
- Provide two or three practical risk-management observations.
- Keep the response concise and structured.
"""

    explanation = ask_gemini(explanation_prompt)

    return {
        "question": question,
        "parameters_applied": parameters,
        "simulation_summary": summary,
        "daily_forecast": daily_forecast,
        "explanation": explanation,
        "scenario_count": len(results),
        "event_count": len(events),
        "data_type": "simulation",
        "configuration_used": {
                "initial_balance": config.initial_balance,
                "daily_expenses": config.daily_expenses,
                "payments_per_day": config.payments_per_day,
                "payment_amount": config.payment_amount,
                "reserve_threshold": config.reserve_threshold,
                "days": config.days,
                "scenarios": config.scenarios,
                "normal_delay_probability": config.normal_delay_probability,
                "stress_delay_probability": config.stress_delay_probability,
                "normal_resolution_probability": config.normal_resolution_probability,
                "stress_resolution_probability": config.stress_resolution_probability,
                "refund_probability": config.refund_probability,
                "regime_model": config.regime_model,
        },
    }
