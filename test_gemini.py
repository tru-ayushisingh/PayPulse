from src.paypulse.ai_client import ask_gemini

response = ask_gemini(
    "Explain what liquidity risk means for a small online merchant "
    "in three simple sentences."
)

print("\nPayPulse AI response:\n")
print(response)