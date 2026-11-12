import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";

import { useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Bot,
  CreditCard,
  LayoutDashboard,
  MessageSquare,
  RefreshCw,
  Send,
  ShieldCheck,
  Wallet,
} from "lucide-react";
import {
  analyzeMerchantQuestion,
} from "./services/api";
import "./App.css";

const formatMoney = (value) =>
  value == null
    ? "—"
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
      }).format(value);

function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysis, setAnalysis] = useState(null);

  const forecastData =
  analysis && config
    ? [
        {
          day: 0,
          average_balance: config.initial_balance,
          lower_bound: config.initial_balance,
          upper_bound: config.initial_balance,
        },
        ...analysis.daily_forecast,
      ]
    : [];

  const runAnalysis = async (event) => {
    event.preventDefault();

    if (!question.trim() || loading) return;

    setLoading(true);
    setError("");
    setAnalysis(null);

    try {
      const result = await analyzeMerchantQuestion(question.trim());
      setAnalysis(result);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Unable to complete the analysis. Check that the API server is running and try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const summary = analysis?.simulation_summary;
  const config = analysis?.configuration_used;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={23} />
          </div>
          <div>
            <h1>PayPulse</h1>
            <span>PAYMENT INTELLIGENCE</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav className="navigation">
          <a className="nav-item active" href="#overview">
            <LayoutDashboard size={18} />
            Overview
          </a>
          <a className="nav-item" href="#ai-analyst">
            <MessageSquare size={18} />
            AI Analyst
          </a>
          <a className="nav-item" href="#risk">
            <ShieldCheck size={18} />
            Risk Analytics
          </a>
          <a className="nav-item" href="#scenarios">
            <Activity size={18} />
            Scenario Lab
          </a>
          <a className="nav-item" href="#paypal">
            <CreditCard size={18} />
            PayPal Integration
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="connection-status">
            <span className="status-dot" />
            API connection configured
          </div>
          <div className="profile">
            <div className="avatar">PP</div>
            <div>
              <strong>Merchant Workspace</strong>
              <span>Sandbox environment</span>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">Workspace / Overview</div>
            <h2>Liquidity Overview</h2>
          </div>
          <div className="topbar-right">
            <span className="environment-tag">
              <span className="status-dot" />
              DEVELOPMENT
            </span>
            <div className="top-avatar">PP</div>
          </div>
        </header>

        <section className="welcome-section" id="overview">
          <div>
            <p className="eyebrow">MERCHANT INTELLIGENCE</p>
            <h1>Know your cash flow before it happens.</h1>
            <p>
              Simulate payment disruptions, understand liquidity exposure,
              and make informed decisions with AI-powered analysis.
            </p>
          </div>
          <div className="welcome-icon">
            <Wallet size={38} />
          </div>
        </section>

        <section className="section-heading">
          <div>
            <h3>Liquidity snapshot</h3>
            <p>
              {summary
                ? "Based on your latest simulation"
                : "Run an AI analysis to populate these metrics"}
            </p>
          </div>
          <span className="live-label">
            <span className="status-dot" />
            SIMULATION DATA
          </span>
        </section>

        <section className="metrics-grid">
          <MetricCard
            title="Starting balance"
            value={formatMoney(config?.initial_balance)}
            icon={<Wallet size={19} />}
            description="Simulation input"
          />
          <MetricCard
            title="Average ending balance"
            value={formatMoney(summary?.average_ending_balance)}
            icon={<ArrowUpRight size={19} />}
            description="Across simulated scenarios"
          />
          <MetricCard
            title="Average minimum balance"
            value={formatMoney(summary?.average_minimum_balance)}
            icon={<ArrowDownRight size={19} />}
            description="Lowest balance per scenario, averaged"
          />
          <MetricCard
            title="Reserve breach probability"
            value={
              summary
                ? `${(summary.reserve_breach_probability * 100).toFixed(2)}%`
                : "—"
            }
            icon={<ShieldCheck size={19} />}
            description="Across simulated scenarios"
          />
        </section>
        
          <section className="panel forecast-panel">
            <div className="panel-heading">
              <div className="heading-icon blue">
                <Activity size={20} />
              </div>
              <div>
                <h3>Cash-flow forecast</h3>
        <p>
          {analysis
            ? `Daily balance projection across ${analysis.scenario_count} simulations`
            : "Run an AI analysis to generate your cash-flow forecast"}
        </p>
      </div>
    </div>

    {forecastData.length > 0 ? (
      <div className="forecast-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={forecastData}
            margin={{ top: 15, right: 20, left: 10, bottom: 5 }}
          >
            <CartesianGrid stroke="#263449" strokeDasharray="3 3" />

            <XAxis
              dataKey="day"
              stroke="#8997ad"
              tickLine={false}
              label={{
                value: "Simulation day",
                position: "insideBottom",
                offset: -2,
                fill: "#8997ad",
                fontSize: 11,
              }}
            />

            <YAxis
              stroke="#8997ad"
              tickLine={false}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
              width={85}
            />

            <Tooltip
              contentStyle={{
                background: "#101827",
                border: "1px solid #35435b",
                borderRadius: "9px",
                color: "#edf3fc",
              }}
              formatter={(value, name) => [
                formatMoney(value),
                name,
              ]}
              labelFormatter={(day) =>
                day === 0 ? "Starting balance" : `Day ${day}`
              }
            />

            <Legend />

            <ReferenceLine
              y={config.reserve_threshold}
              stroke="#f87171"
              strokeDasharray="6 4"
              label={{
                value: "Reserve threshold",
                fill: "#f87171",
                fontSize: 10,
              }}
            />

            <Line
              type="monotone"
              dataKey="average_balance"
              name="Average balance"
              stroke="#a78bfa"
              strokeWidth={3}
              dot={false}
            />

            <Line
              type="monotone"
              dataKey="lower_bound"
              name="10th percentile"
              stroke="#60a5fa"
              strokeWidth={1.5}
              strokeDasharray="5 4"
              dot={false}
            />

            <Line
              type="monotone"
              dataKey="upper_bound"
              name="90th percentile"
              stroke="#34d399"
              strokeWidth={1.5}
              strokeDasharray="5 4"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    ) : (
      <div className="forecast-empty">
        <Activity size={28} />
        <p>Your daily cash-flow forecast will appear here after analysis.</p>
      </div>
    )}
  </section>


        <section className="content-grid">
          <div className="panel ai-panel" id="ai-analyst">
            <div className="panel-heading">
              <div className="heading-icon purple">
                <Bot size={20} />
              </div>
              <div>
                <h3>PayPulse AI Analyst</h3>
                <p>Ask questions about your business liquidity</p>
              </div>
              <span className="ai-badge">GEMINI POWERED</span>
            </div>

            <div className="chat-intro">
              <div className="bot-avatar">
                <Bot size={23} />
              </div>
              <div>
                <strong>What would you like to investigate?</strong>
                <p>
                  Ask me to simulate payment delays, refunds, cash-flow
                  disruptions, or reserve requirements.
                </p>
              </div>
            </div>

            <div className="suggestion-list">
              {[
                "What happens if 20% of payments are delayed?",
                "Can my business survive 30 days of higher expenses?",
                "How would persistent payment disruptions affect my liquidity?",
              ].map((suggestion) => (
                <button
                  className="suggestion"
                  key={suggestion}
                  onClick={() => setQuestion(suggestion)}
                  type="button"
                >
                  {suggestion}
                </button>
              ))}
            </div>

            <form className="chat-form" onSubmit={runAnalysis}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask PayPulse about your liquidity risk..."
                rows={3}
                maxLength={2000}
              />
              <div className="chat-form-bottom">
                <span>{question.length}/2000 characters</span>
                <button
                  className="primary-button"
                  type="submit"
                  disabled={loading || question.trim().length < 5}
                >
                  {loading ? (
                    <>
                      <RefreshCw size={16} className="spin" />
                      Analysing...
                    </>
                  ) : (
                    <>
                      Run analysis
                      <Send size={16} />
                    </>
                  )}
                </button>
              </div>
            </form>

            {loading && (
              <div className="loading-message">
                Gemini is interpreting your question and running the
                simulation. This may take a little while.
              </div>
            )}

            {error && <div className="error-message">{error}</div>}

            {analysis && (
              <div className="analysis-result">
                <div className="result-title">
                  <ShieldCheck size={19} />
                  Simulation completed
                </div>

                <div className="result-stats">
                  <div>
                    <span>Ending balance</span>
                    <strong>
                      {formatMoney(summary?.average_ending_balance)}
                    </strong>
                  </div>
                  <div>
                    <span>Reserve breach</span>
                    <strong>
                      {summary
                        ? `${(summary.reserve_breach_probability * 100).toFixed(2)}%`
                        : "—"}
                    </strong>
                  </div>
                  <div>
                    <span>Simulations</span>
                    <strong>
                      {analysis.scenario_count?.toLocaleString()}
                    </strong>
                  </div>
                </div>

                <div className="explanation">
                  <h4>AI risk assessment</h4>
                  <p>{analysis.explanation}</p>
                </div>

                <div className="assumptions">
                  <h4>Configuration used</h4>
                  <div className="assumption-grid">
                    {Object.entries(config || {}).map(([key, value]) => (
                      <div className="assumption" key={key}>
                        <span>{key.replaceAll("_", " ")}</span>
                        <strong>
                          {typeof value === "number"
                            ? value.toLocaleString()
                            : String(value)}
                        </strong>
                      </div>
                    ))}
                  </div>
                </div>

                <p className="simulation-disclaimer">
                  Simulated estimates, not verified PayPal settlements or
                  guaranteed future outcomes.
                </p>
              </div>
            )}
          </div>

          <div className="right-column">
            <div className="panel" id="risk">
              <div className="panel-heading compact">
                <div className="heading-icon green">
                  <ShieldCheck size={19} />
                </div>
                <div>
                  <h3>Risk analytics</h3>
                  <p>Liquidity risk indicators</p>
                </div>
              </div>

              <RiskMetric
                label="Liquidity VaR (95%)"
                value={formatMoney(summary?.liquidity_var_95)}
              />
              <RiskMetric
                label="Liquidity CVaR (95%)"
                value={formatMoney(summary?.liquidity_cvar_95)}
              />
              <RiskMetric
                label="Average reserve shortfall"
                value={formatMoney(summary?.average_reserve_shortfall)}
              />
              <RiskMetric
                label="Outstanding payment amount"
                value={formatMoney(summary?.average_outstanding_payment_amount)}
              />
              <RiskMetric
                label="Outstanding payment count"
                value={
                  summary?.average_outstanding_payment_count?.toFixed(2) ?? "—"
                }
              />
              <p className="muted-note">
                Metrics are populated from the most recent AI simulation.
              </p>
            </div>

            <div className="panel scenario-panel" id="scenarios">
              <div className="panel-heading compact">
                <div className="heading-icon blue">
                  <Activity size={19} />
                </div>
                <div>
                  <h3>Scenario lab</h3>
                  <p>Explore operational what-if questions</p>
                </div>
              </div>
              <div className="scenario-empty">
                <Activity size={27} />
                <strong>Compare business scenarios</strong>
                <p>
                  Use the AI Analyst to explore normal operations and
                  persistent disruption scenarios.
                </p>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setQuestion(
                      "Compare normal operations with persistent payment disruption over 10 days."
                    );
                    document
                      .getElementById("ai-analyst")
                      ?.scrollIntoView({ behavior: "smooth" });
                  }}
                >
                  Explore with AI
                </button>
              </div>
            </div>

            <div className="panel paypal-panel" id="paypal">
              <div className="panel-heading compact">
                <div className="heading-icon blue">
                  <CreditCard size={19} />
                </div>
                <div>
                  <h3>PayPal connection</h3>
                  <p>Sandbox integration</p>
                </div>
              </div>
              <div className="paypal-status">
                <span className="status-dot" />
                Order API configured
              </div>
              <p className="muted-note">
                Sandbox order creation is integrated. Buyer approval and
                capture have not yet been verified.
              </p>
            </div>
          </div>
        </section>

        <footer className="footer">
          <span>PayPulse · Counterfactual Payment Intelligence</span>
          <span>AI-assisted analysis · Simulation-based estimates</span>
        </footer>
      </main>
    </div>
  );
}

function MetricCard({ title, value, icon, description }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <div className="metric-icon">{icon}</div>
      </div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-description">{description}</span>
    </div>
  );
}

function RiskMetric({ label, value }) {
  return (
    <div className="risk-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
