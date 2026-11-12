
from fastapi.testclient import TestClient

from src.paypulse.api import app

from unittest.mock import patch

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_simulation_endpoint():
    response = client.post(
        "/api/simulate",
        json={
            "initial_balance": 6000,
            "daily_expenses": 650,
            "payments_per_day": 10,
            "payment_amount": 60,
            "reserve_threshold": 5600,
            "days": 7,
            "scenarios": 100,
            "delay_probability": 0.08,
            "refund_probability": 0.08,
            "minimum_refund": 100,
            "maximum_refund": 250,
            "seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "summary" in data
    assert "chart" in data
    assert len(data["chart"]) == 8


def test_invalid_probability():
    response = client.post(
        "/api/simulate",
        json={"delay_probability": 1.5},
    )

    assert response.status_code == 422


def test_compare_endpoint():
    response = client.post(
        "/api/compare",
        json={
            "base": {
                "scenarios": 100,
                "seed": 42,
            },
            "stress_delay_probability": 0.25,
            "stress_refund_probability": 0.20,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "baseline" in data
    assert "stress" in data
    assert "risk_difference" in data



def test_backlog_simulation_endpoint():
    response = client.post(
        "/api/backlog/simulate",
        json={
            "initial_balance": 6000,
            "daily_expenses": 650,
            "payments_per_day": 5,
            "payment_amount": 60,
            "reserve_threshold": 5600,
            "days": 3,
            "scenarios": 100,
            "normal_delay_probability": 0.05,
            "stress_delay_probability": 0.50,
            "refund_probability": 0.08,
            "seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "summary" in data
    assert "chart" in data
    assert "events" in data
    assert "event_count" in data
    assert len(data["chart"]) == 3

    assert "reserve_breach_probability" in data["summary"]
    assert "liquidity_cvar_95" in data["summary"]


def test_backlog_comparison_endpoint():
    response = client.post(
        "/api/backlog/compare",
        json={
            "scenarios": 100,
            "days": 3,
            "payments_per_day": 5,
            "seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "independent" in data
    assert "markov" in data
    assert "differences" in data

    assert "breach_probability" in data["differences"]
    assert "average_shortfall" in data["differences"]
    assert "liquidity_cvar_95" in data["differences"]


def test_invalid_backlog_refund_fraction():
    response = client.post(
        "/api/backlog/simulate",
        json={
            "minimum_refund_fraction": 0.8,
            "maximum_refund_fraction": 0.5,
        },
    )

    assert response.status_code == 422


def test_invalid_backlog_regime():
    response = client.post(
        "/api/backlog/simulate",
        json={"regime_model": "unknown"},
    )

    assert response.status_code == 422


def test_paypal_create_order_endpoint():
    mock_order = {
        "order_id": "TESTORDER123",
        "status": "PAYER_ACTION_REQUIRED",
        "approval_url": "https://www.sandbox.paypal.com/checkoutnow?token=TESTORDER123",
        "amount": "25.00",
        "currency": "USD",
    }

    with patch(
        "src.paypulse.api.create_paypal_order",
        return_value=mock_order,
    ):
        response = client.post(
            "/api/paypal/orders",
            json={
                "amount": "25.00",
                "currency": "USD",
                "description": "PayPulse test payment",
            },
        )

    assert response.status_code == 200
    data = response.json()

    assert data["order_id"] == "TESTORDER123"
    assert data["amount"] == "25.00"
    assert data["currency"] == "USD"
    assert "approval_url" in data


def test_paypal_capture_order_endpoint():
    mock_capture = {
        "order_id": "TESTORDER123",
        "order_status": "COMPLETED",
        "capture_id": "TESTCAPTURE123",
        "capture_status": "COMPLETED",
        "amount": "25.00",
        "currency": "USD",
    }

    with patch(
        "src.paypulse.api.capture_paypal_order",
        return_value=mock_capture,
    ):
        response = client.post(
            "/api/paypal/orders/TESTORDER123/capture"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["payment_completed"] is True
    assert data["capture_status"] == "COMPLETED"


def test_paypal_capture_does_not_assume_completion():
    mock_capture = {
        "order_id": "TESTORDER123",
        "order_status": "COMPLETED",
        "capture_id": "TESTCAPTURE123",
        "capture_status": "PENDING",
        "amount": "25.00",
        "currency": "USD",
    }

    with patch(
        "src.paypulse.api.capture_paypal_order",
        return_value=mock_capture,
    ):
        response = client.post(
            "/api/paypal/orders/TESTORDER123/capture"
        )

    assert response.status_code == 200
    assert response.json()["payment_completed"] is False


def test_paypal_order_rejects_invalid_amount():
    response = client.post(
        "/api/paypal/orders",
        json={"amount": "-10.00"},
    )

    assert response.status_code == 422


def test_paypal_capture_rejects_invalid_order_id():
    response = client.post(
        "/api/paypal/orders/invalid-order/capture"
    )

    assert response.status_code == 422

def test_ai_analyze_endpoint():
    mock_result = {
        "question": "What if 20% of payments are delayed?",
        "parameters_applied": {
            "stress_delay_probability": 0.20
        },
        "simulation_summary": {
            "reserve_breach_probability": 0.25,
            "liquidity_cvar_95": 500.0,
        },
        "explanation": "The simulated scenario shows elevated liquidity risk.",
        "scenario_count": 1000,
        "event_count": 100,
        "data_type": "simulation",
    }

    with patch(
        "src.paypulse.api.run_ai_analysis",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/ai/analyze",
            json={
                "question": "What if 20% of payments are delayed?"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["parameters_applied"]["stress_delay_probability"] == 0.20
    assert "simulation_summary" in data
    assert "explanation" in data
    assert data["data_type"] == "simulation"


def test_ai_analyze_rejects_short_question():
    response = client.post(
        "/api/ai/analyze",
        json={"question": "Hi"},
    )

    assert response.status_code == 422

