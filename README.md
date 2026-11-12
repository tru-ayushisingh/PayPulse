PayPulse — Counterfactual Payment Intelligence
PayPulse is an AI-assisted merchant payment and liquidity risk analysis prototype. It helps merchants explore operational “what if?” questions—such as delayed payments, refunds, and persistent payment disruptions—and understand how those scenarios could affect short-term cash availability.

Rather than asking an AI model to invent financial forecasts, PayPulse uses AI to translate a natural-language question into validated simulation parameters, runs a quantitative simulation engine, and then uses AI to explain the computed results.
Hackathon prototype: PayPulse uses synthetic merchant assumptions and PayPal Sandbox integration. Simulated cash flows are not PayPal settlement records, bank balances, or financial advice.

The problem
Merchants can experience cash-flow pressure when incoming payments are delayed, refunds increase, or payment operations are disrupted. A single expected-value forecast can hide the impact of timing, correlated events, and severe downside scenarios.

PayPulse explores these risks through configurable simulations and communicates the results in a merchant-friendly interface.
What PayPulse does
- Natural-language analysis: Ask the AI Analyst a merchant liquidity “what if?” question.
- Parameter extraction: Gemini extracts candidate simulation settings; the backend validates them before execution.
- Quantitative simulation: A Python engine models payment events, delays, refunds, outstanding payment backlogs, and disruption regimes.
- Scenario comparison: API support for baseline and scenario comparisons, including independent and persistent/Markov-style disruption assumptions.
- Downside-risk measurement: Liquidity summaries and risk measures such as reserve-breach probability, Value at Risk (VaR), Conditional Value at Risk (CVaR), and outstanding-backlog metrics where applicable.
- Liquidity visualization: A React dashboard presents simulation outputs and a daily cash-balance forecast.
- PayPal Sandbox integration: The backend authenticates with PayPal Sandbox and creates checkout orders. A capture endpoint is implemented, but a completed buyer approval/capture flow has not been verified for this demo.
How it works
1. A merchant enters a question in the React dashboard.
2. Gemini extracts a constrained set of simulation parameters.
3. FastAPI validates those parameters and builds a simulation configuration.
4. The Python engine runs the requested scenarios using synthetic assumptions.
5. The backend aggregates the output into summary metrics and a daily forecast.
6. Gemini explains the computed results. Numerical results come from the simulation engine, not from the language model.
Architecture
React + Vite dashboard
        |
        | HTTP / Axios
        v
FastAPI backend
   |           |
   |           +------ Gemini API
   |                  - Question-to-parameter extraction
   |                  - Explanation of computed results
   |
   +------ PayPulse simulation engine
   |       - Payment events and delays
   |       - Refunds and outstanding backlog
   |       - Disruption regimes
   |       - Monte Carlo scenarios
   |       - Liquidity risk metrics
   |
   +------ PayPal REST API (Sandbox)
           - OAuth access token
           - Create checkout order
           - Capture endpoint
Technology stack
Area	Technologies
Frontend	React, Vite, JavaScript, Tailwind CSS, Recharts, Axios, Lucide
Backend	Python, FastAPI, Pydantic, Uvicorn
Simulation and analysis	NumPy, Pandas, Monte Carlo simulation, event-driven modelling, VaR, CVaR
AI	Google Gemini API (google-genai)
Payments integration	PayPal REST API — Sandbox
Testing	pytest


Quantitative model
PayPulse models merchant cash balance as a function of inflows and outflows:
\[
B_t = B_0 + \sum_{i=1}^{t}(R_i-E_i)
\]
where:
- \(B_0\) is the initial balance;
- \(R_i\) represents simulated incoming receipts;
- \(E_i\) represents simulated expenses and other outflows.
The simulation can incorporate delayed payment events, refunds, unresolved payment backlogs, and disruption regimes. Multiple scenarios are run to estimate a distribution of possible outcomes rather than relying on a single path.

Risk measures are interpreted relative to the configured reserve threshold. VaR and CVaR describe downside outcomes under the model's assumptions; they are not guarantees or predictions of actual merchant losses.

Modelling limitation: Inputs and event distributions are prototype assumptions. Results are scenario-analysis outputs, not calibrated forecasts based on a merchant's historical transaction data.
PayPal integration status
PayPulse integrates with the PayPal Sandbox, not live merchant accounts.

Implemented:
- Sandbox OAuth 2.0 client-credentials authentication.
- Create a Sandbox checkout order (intent: CAPTURE).
- Capture-order API endpoint.
Verified during development:
- Sandbox authentication succeeded.
- Order creation returned HTTP 200 with an order ID, CREATED status, and an approval URL.
Not verified:
- End-to-end buyer approval followed by a successful capture with COMPLETED status. Sandbox checkout attempts encountered an international-regulations transaction decline.
Therefore, this project does not claim that a payment was successfully captured, settled, or made available in a merchant's bank account. Simulation receipts are synthetic modelled events and are separate from PayPal transaction or settlement data.
Run locally
Prerequisites
- Python 3.11+ recommended.
- Node.js and npm.
- Git.
- Gemini API key.
- PayPal Sandbox REST app credentials (optional if you are not testing PayPal endpoints).
1. Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/PayPulse.git
cd PayPulse
Replace YOUR_GITHUB_USERNAME with the GitHub account that owns the repository.
2. Create and activate a Python environment
Windows PowerShell:
py -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks activation, use the environment's Python directly:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Otherwise install dependencies:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
3. Configure environment variables
Create a .env file in the repository root. Do not commit this file.
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite

PAYPAL_CLIENT_ID=your_paypal_sandbox_client_id
PAYPAL_CLIENT_SECRET=your_paypal_sandbox_secret
PAYPAL_BASE_URL=https://api-m.sandbox.paypal.com
The Gemini key is required for AI analysis. PayPal credentials are required only for PayPal Sandbox endpoints.
4. Start the backend
From the repository root:
python -m uvicorn src.paypulse.api:app --reload
The API runs at http://127.0.0.1:8000.
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
5. Start the frontend
Open a second terminal:
cd frontend
npm install
npm run dev
Open the Vite URL shown in the terminal, normally http://localhost:5173.

For a deployed frontend, set VITE_API_BASE_URL to the public FastAPI service URL before building.
API overview
The FastAPI interactive documentation at /docs is the source of truth for request schemas and available fields.
Endpoint	Purpose
GET /	API root/status
GET /health	Health check
POST /api/simulate	Run the original payment simulation
POST /api/compare	Compare simulation configurations
POST /api/backlog/simulate	Run event/backlog-based simulation
POST /api/backlog/compare	Compare backlog/disruption scenarios
POST /api/paypal/orders	Create a PayPal Sandbox order
POST /api/paypal/orders/{order_id}/capture	Request capture of a Sandbox order
AI analysis endpoint	Accepts a natural-language merchant question and returns extracted parameters, simulation results, and an explanation. See /docs for the exact route and schema.


Tests
Run the test suite from the repository root:
python -m pytest -v
The project test suite was run during development. Run the command above against the submitted commit to reproduce the result in your environment.
Deployment
The application can be deployed as two services.
Backend — Render
- Runtime: Python.
- Root directory: repository root (leave blank if requirements.txt is at the repository root).
- Build command:
pip install -r requirements.txt
- Start command:
uvicorn src.paypulse.api:app --host 0.0.0.0 --port $PORT
- Configure GEMINI_API_KEY, GEMINI_MODEL, and the PayPal Sandbox environment variables in Render service settings.
- Verify https://YOUR-RENDER-SERVICE.onrender.com/health after deployment.
Frontend — Vercel
- Framework preset: Vite.
- Root directory: frontend.
- Build command: npm run build.
- Output directory: dist.
- Environment variable:
VITE_API_BASE_URL=https://YOUR-RENDER-SERVICE.onrender.com
- Add the deployed Vercel origin to FastAPI CORS allow_origins, alongside the local development origin.
- Redeploy the backend after changing CORS.
Replace example domains with actual deployment URLs. Never publish API keys or Sandbox secrets in the repository.
Project structure
PayPulse/
├── src/
│   └── paypulse/
│       ├── api.py
│       ├── ai_agent.py
│       ├── ai_client.py
│       ├── backlog_simulation.py
│       ├── event_simulation.py
│       ├── paypal_client.py
│       ├── regimes.py
│       ├── research.py
│       ├── risk_metrics.py
│       ├── simulation.py
│       └── visualization.py
├── tests/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── requirements.txt
├── .env.example
└── README.md
Research and experiment scripts may also be present in the repository root.
Assumptions, limitations, and responsible use
- PayPulse is a hackathon research prototype, not a production treasury, accounting, payment-processing, or investment system.
- Simulation outputs depend on user-supplied or default assumptions and random scenario generation.
- Synthetic payment receipts must not be interpreted as actual PayPal settlement or bank-available funds.
- The model does not establish that a specific merchant will experience a particular loss or reserve breach.
- Gemini parameter extraction is constrained and validated by the backend, but AI-generated explanations should still be reviewed.
- PayPal functionality uses Sandbox credentials and test transactions only.
- Do not add real customer payment credentials or private transaction data to the public repository.
Demo
- Hosted application: ADD_YOUR_DEPLOYED_FRONTEND_URL
- Demo video: ADD_YOUR_PUBLIC_YOUTUBE_DEMO_URL
- Source repository: https://github.com/YOUR_GITHUB_USERNAME/PayPulse
In the demo, show a natural-language merchant question, the applied configuration, simulation summary, liquidity forecast/risk metrics, and PayPal Sandbox order-creation capability. State clearly that end-to-end capture was not verified.
License
This project is released under the MIT License. See LICENSE.
