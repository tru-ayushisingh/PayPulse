
import os

import uuid
from decimal import Decimal, ROUND_HALF_UP

import requests
from dotenv import load_dotenv

# Load credentials from the project's .env file.
load_dotenv()

class PayPalAPIError(Exception):
    """Represents an error returned by the PayPal API."""

    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def get_paypal_config():
    """Load and validate PayPal Sandbox configuration."""

    client_id = os.getenv("PAYPAL_CLIENT_ID")
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
    base_url = os.getenv(
        "PAYPAL_BASE_URL",
        "https://api-m.sandbox.paypal.com",
    )

    if not client_id or not client_secret:
        raise ValueError(
            "PayPal credentials are missing. Check your .env file."
        )

    if not base_url.startswith("https://"):
        raise ValueError("PayPal API URL must use HTTPS.")

    return client_id, client_secret, base_url.rstrip("/")


def get_access_token():
    """Obtain an OAuth 2.0 access token from PayPal."""

    client_id, client_secret, base_url = get_paypal_config()

    url = f"{base_url}/v1/oauth2/token"

    try:
        response = requests.post(
            url,
            auth=(client_id, client_secret),
            headers={
                "Accept": "application/json",
                "Accept-Language": "en_US",
            },
            data={"grant_type": "client_credentials"},
            timeout=20,
        )

        if not response.ok:
            raise RuntimeError(
                f"PayPal authentication failed with HTTP "
                f"{response.status_code}."
            )

        data = response.json()
        access_token = data.get("access_token")

        if not access_token:
            raise RuntimeError(
                "PayPal response did not contain an access token."
            )

        return access_token

    except requests.RequestException as error:
        raise RuntimeError(
            "Could not connect to PayPal Sandbox."
        ) from error

def paypal_request(method, endpoint, payload=None):
    """Make an authenticated request to the PayPal REST API."""

    _, _, base_url = get_paypal_config()
    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "return=representation",
        "PayPal-Request-Id": str(uuid.uuid4()),
    }

    try:
        response = requests.request(
            method=method,
            url=f"{base_url}{endpoint}",
            headers=headers,
            json=payload,
            timeout=30,
        )

        try:
            data = response.json()
        except ValueError:
            data = {}

        if not response.ok:
            message = data.get(
                "message",
                data.get("name", "PayPal API request failed."),
            )

            raise PayPalAPIError(
                message,
                response.status_code,
            )

        return data

    except requests.RequestException as error:
        raise PayPalAPIError(
            "Could not communicate with PayPal."
        ) from error


def create_paypal_order(amount, currency="USD", description="PayPulse test payment"):
    """Create a PayPal order and return its buyer approval URL."""

    amount = Decimal(str(amount)).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    payload = {
        "intent": "CAPTURE",
        "purchase_units": [
            {
                "description": description,
                "amount": {
                    "currency_code": currency,
                    "value": str(amount),
                },
            }
        ],
    }

    data = paypal_request(
        "POST",
        "/v2/checkout/orders",
        payload,
    )

    approval_url = next(
        (
            link["href"]
            for link in data.get("links", [])
            if link.get("rel") in ("payer-action", "approve")
        ),
        None,
    )

    if not data.get("id"):
        raise PayPalAPIError("PayPal did not return an order ID.")

    if not approval_url:
        raise PayPalAPIError(
            "PayPal did not return a buyer approval URL."
        )

    return {
        "order_id": data["id"],
        "status": data.get("status"),
        "approval_url": approval_url,
        "amount": str(amount),
        "currency": currency,
    }


def capture_paypal_order(order_id):
    """Capture an order after the buyer has approved it."""

    data = paypal_request(
        "POST",
        f"/v2/checkout/orders/{order_id}/capture",
        {},
    )

    captures = []

    for purchase_unit in data.get("purchase_units", []):
        payments = purchase_unit.get("payments", {})
        captures.extend(payments.get("captures", []))

    first_capture = captures[0] if captures else {}

    return {
        "order_id": data.get("id"),
        "order_status": data.get("status"),
        "capture_id": first_capture.get("id"),
        "capture_status": first_capture.get("status"),
        "amount": first_capture.get("amount", {}).get("value"),
        "currency": first_capture.get("amount", {}).get("currency_code"),
    }

