
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


class GeminiAPIError(Exception):
    """Raised when Gemini cannot process a request."""


def get_gemini_client():
    """Create a Gemini client using the API key in .env."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Check your .env file."
        )

    return genai.Client(api_key=api_key)



def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini with model fallback."""

    primary_model = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )

    fallback_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash",
    ]

    models_to_try = list(dict.fromkeys(
        [primary_model] + fallback_models
    ))

    client = get_gemini_client()
    last_error = None

    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "system_instruction": (
                        "You are PayPulse AI, a fintech liquidity "
                        "intelligence assistant. Explain financial "
                        "simulation results accurately. Never invent "
                        "numerical results or financial facts. "
                        "Distinguish simulations from verified "
                        "PayPal events."
                    ),
                    "temperature": 0.2,
                    "max_output_tokens": 800,
                },
            )

            if not response.text:
                raise GeminiAPIError(
                    "Gemini returned an empty response."
                )

            print(f"Gemini response generated using: {model}")
            return response.text

        except GeminiAPIError:
            raise

        except Exception as error:
            last_error = error
            status_code = getattr(error, "status_code", None)

            if status_code != 503:
                break

    raise GeminiAPIError(
        "Gemini is temporarily unavailable. "
        "Check model availability and API quota. "
        f"Last error: {last_error}"
    ) from last_error

