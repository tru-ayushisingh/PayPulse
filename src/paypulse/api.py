
import re
from decimal import Decimal
from fastapi import FastAPI, HTTPException, Path
from dataclasses import replace
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.paypulse.ai_agent import run_ai_analysis
from src.paypulse.ai_client import GeminiAPIError
from fastapi.middleware.cors import CORSMiddleware

from src.paypulse.paypal_client import (
    create_paypal_order,
    capture_paypal_order,
    PayPalAPIError,
)

from src.paypulse.simulation import (
    SimulationConfig,
    run_simulation,
    summarize_results,
)
from src.paypulse.backlog_simulation import (
    BacklogSimulationConfig,
    run_backlog_simulation,
    summarize_backlog_results,
)


app = FastAPI(
    title="PayPulse API",
    description="Quantitative payment and liquidity simulation engine",
    version="0.2.0",
)

# Enable communication with React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# ORIGINAL AGGREGATE SIMULATION REQUESTS
# --------------------------------------------------

class SimulationRequest(BaseModel):
    initial_balance: float = Field(default=6000, ge=0)
    daily_expenses: float = Field(default=650, ge=0)
    payments_per_day: int = Field(default=10, ge=0, le=1000)
    payment_amount: float = Field(default=60, ge=0)
    reserve_threshold: float = Field(default=5600, ge=0)
    days: int = Field(default=7, ge=1, le=90)
    scenarios: int = Field(default=5000, ge=100, le=50000)
    delay_probability: float = Field(default=0.08, ge=0, le=1)
    refund_probability: float = Field(default=0.08, ge=0, le=1)
    minimum_refund: float = Field(default=100, ge=0)
    maximum_refund: float = Field(default=250, ge=0)
    seed: int = Field(default=42, ge=0)


class CompareRequest(BaseModel):
    base: SimulationRequest
    stress_delay_probability: float = Field(default=0.25, ge=0, le=1)
    stress_refund_probability: float = Field(default=0.20, ge=0, le=1)

class AIAnalyzeRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)


def build_config(request: SimulationRequest) -> SimulationConfig:
    return SimulationConfig(**request.model_dump())


def execute_scenario(config: SimulationConfig) -> dict:
    daily, results = run_simulation(config)
    summary = summarize_results(config, results)

    chart = []
    for day in range(config.days + 1):
        values = daily[f"day_{day}"]
        chart.append({
            "day": day,
            "p05": round(float(values.quantile(0.05)), 2),
            "p50": round(float(values.quantile(0.50)), 2),
            "p95": round(float(values.quantile(0.95)), 2),
        })

    return {"summary": summary, "chart": chart}


# --------------------------------------------------
# EVENT-DRIVEN BACKLOG SIMULATION REQUESTS
# --------------------------------------------------

class BacklogSimulationRequest(BaseModel):
    initial_balance: float = Field(default=6000, ge=0)
    daily_expenses: float = Field(default=650, ge=0)

    payments_per_day: int = Field(default=10, ge=0, le=100)
    payment_amount: float = Field(default=60, ge=0)

    reserve_threshold: float = Field(default=5600, ge=0)
    days: int = Field(default=7, ge=1, le=90)
    scenarios: int = Field(default=1000, ge=100, le=5000)

    normal_delay_probability: float = Field(default=0.05, ge=0, le=1)
    stress_delay_probability: float = Field(default=0.50, ge=0, le=1)

    normal_resolution_probability: float = Field(default=0.75, ge=0, le=1)
    stress_resolution_probability: float = Field(default=0.25, ge=0, le=1)

    refund_probability: float = Field(default=0.08, ge=0, le=1)
    minimum_refund_fraction: float = Field(default=0.10, ge=0, le=1)
    maximum_refund_fraction: float = Field(default=0.50, ge=0, le=1)

    regime_model: str = Field(default="markov", pattern="^(independent|markov)$")

    disruption_probability: float = Field(default=0.08, ge=0, le=1)
    normal_to_stress: float = Field(default=0.04, ge=0, le=1)
    stress_to_normal: float = Field(default=0.46, ge=0, le=1)

    seed: int = Field(default=42, ge=0)


def build_backlog_config(
    request: BacklogSimulationRequest,
) -> BacklogSimulationConfig:
    return BacklogSimulationConfig(**request.model_dump())


def execute_backlog_scenario(config: BacklogSimulationConfig) -> dict:
    daily, results, events = run_backlog_simulation(config)
    summary = summarize_backlog_results(config, results)

    # Daily balance and backlog percentiles for frontend charts.
    chart = []
    for day, group in daily.groupby("day"):
        chart.append({
            "day": int(day),
            "balance_p05": round(float(group["balance"].quantile(0.05)), 2),
            "balance_p50": round(float(group["balance"].quantile(0.50)), 2),
            "balance_p95": round(float(group["balance"].quantile(0.95)), 2),
            "outstanding_amount_p50": round(
                float(group["outstanding_payment_amount"].quantile(0.50)), 2
            ),
            "outstanding_amount_p95": round(
                float(group["outstanding_payment_amount"].quantile(0.95)), 2
            ),
        })

    # Limit response size while retaining the total event count.
    event_limit = 500
    event_records = events.head(event_limit).to_dict(orient="records")

    return {
        "summary": summary,
        "chart": chart,
        "events": event_records,
        "event_count": len(events),
        "events_truncated": len(events) > event_limit,
    }


# --------------------------------------------------
# GENERAL ENDPOINTS
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Welcome to PayPulse API",
        "status": "running",
        "version": "0.2.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# --------------------------------------------------
# ORIGINAL SIMULATION ENDPOINTS
# --------------------------------------------------

@app.post("/api/simulate")
def simulate(request: SimulationRequest):
    try:
        config = build_config(request)
        return execute_scenario(config)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/api/compare")
def compare(request: CompareRequest):
    try:
        base_config = build_config(request.base)

        stress_config = replace(
            base_config,
            delay_probability=request.stress_delay_probability,
            refund_probability=request.stress_refund_probability,
        )

        baseline = execute_scenario(base_config)
        stress = execute_scenario(stress_config)

        baseline_risk = baseline["summary"]["reserve_breach_probability"]
        stress_risk = stress["summary"]["reserve_breach_probability"]

        return {
            "baseline": baseline,
            "stress": stress,
            "risk_difference": round(stress_risk - baseline_risk, 4),
            "interpretation": (
                "Stress conditions increased reserve breach risk."
                if stress_risk > baseline_risk
                else "Stress conditions did not increase reserve breach risk."
            ),
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


# --------------------------------------------------
# NEW EVENT-DRIVEN BACKLOG ENDPOINTS
# --------------------------------------------------

@app.post("/api/backlog/simulate")
def simulate_backlog(request: BacklogSimulationRequest):
    try:
        config = build_backlog_config(request)
        return execute_backlog_scenario(config)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/api/backlog/compare")
def compare_backlog(request: BacklogSimulationRequest):
    try:
        base_config = build_backlog_config(request)

        independent_config = replace(
            base_config,
            regime_model="independent",
        )
        markov_config = replace(
            base_config,
            regime_model="markov",
        )

        independent = execute_backlog_scenario(independent_config)
        markov = execute_backlog_scenario(markov_config)

        independent_summary = independent["summary"]
        markov_summary = markov["summary"]

        return {
            "independent": independent,
            "markov": markov,
            "differences": {
                "breach_probability": round(
                    markov_summary["reserve_breach_probability"]
                    - independent_summary["reserve_breach_probability"],
                    4,
                ),
                "average_shortfall": round(
                    markov_summary["average_reserve_shortfall"]
                    - independent_summary["average_reserve_shortfall"],
                    2,
                ),
                "liquidity_cvar_95": round(
                    markov_summary["liquidity_cvar_95"]
                    - independent_summary["liquidity_cvar_95"],
                    2,
                ),
            },
            "comparison_note": (
                "These are simulated results under the supplied assumptions. "
                "The two regime models use separate random streams, so this "
                "comparison is not a paired common-random-numbers experiment."
            ),
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    
# --------------------------------------------------
# PAYPAL SANDBOX CHECKOUT
# --------------------------------------------------

class PayPalOrderRequest(BaseModel):
    amount: Decimal = Field(
        gt=0,
        max_digits=10,
        decimal_places=2,
    )
    currency: str = Field(default="USD", pattern="^[A-Z]{3}$")
    description: str = Field(
        default="PayPulse Sandbox test payment",
        min_length=1,
        max_length=127,
    )


@app.post("/api/paypal/orders")
def create_order_endpoint(request: PayPalOrderRequest):
    """Create a PayPal Sandbox order."""

    try:
        order = create_paypal_order(
            amount=request.amount,
            currency=request.currency,
            description=request.description,
        )

        return {
            "message": "PayPal order created. Buyer approval is required.",
            **order,
        }

    except PayPalAPIError as error:
        status_code = (
            error.status_code
            if 400 <= error.status_code < 500
            else 502
        )
        raise HTTPException(
            status_code=status_code,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )


@app.post("/api/paypal/orders/{order_id}/capture")
def capture_order_endpoint(
    order_id: str = Path(
        min_length=1,
        max_length=36,
        pattern="^[A-Z0-9]+$",
    ),
):
    """Capture a PayPal order after buyer approval."""

    try:
        capture = capture_paypal_order(order_id)

        completed = capture.get("capture_status") == "COMPLETED"

        return {
            "message": (
                "Payment captured successfully."
                if completed
                else "Capture response received; payment is not confirmed as completed."
            ),
            "payment_completed": completed,
            **capture,
        }

    except PayPalAPIError as error:
        status_code = (
            error.status_code
            if 400 <= error.status_code < 500
            else 502
        )
        raise HTTPException(
            status_code=status_code,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

@app.post("/api/ai/analyze")
def analyze_merchant_question(request: AIAnalyzeRequest):
    try:
        return run_ai_analysis(request.question)

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except GeminiAPIError as error:
        raise HTTPException(
            status_code=502,
            detail=str(error),
        )

