"""
AI Manufacturing Decision Lab
Capstone: Managing an AI-enabled manufacturing enterprise

Run:
    pip install -r requirements.txt
    streamlit run app.py

Optional generative AI:
    Set OPENAI_API_KEY in your environment or enter it in the app's sidebar.
    The core simulation and report templates work without an API key.
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AI Manufacturing Decision Lab",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Configuration ----------------------------
MONTHS = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]
DEFAULT_HISTORY = np.array([940, 1020, 980, 1110, 1060, 1190, 1140, 1210, 1260, 1180, 1320, 1280])
DEFAULT_SUPPLIERS = pd.DataFrame([
    {"Supplier": "Apex Components", "Share (%)": 45, "Lead time (days)": 18, "Quality (%)": 96.5, "Financial health (1-5)": 4, "Geopolitical exposure (1-5)": 2, "Flexibility (1-5)": 3, "Unit cost ($)": 42},
    {"Supplier": "Nova Materials", "Share (%)": 35, "Lead time (days)": 28, "Quality (%)": 94.0, "Financial health (1-5)": 3, "Geopolitical exposure (1-5)": 4, "Flexibility (1-5)": 2, "Unit cost ($)": 45},
    {"Supplier": "Zenith Industrial", "Share (%)": 20, "Lead time (days)": 12, "Quality (%)": 98.0, "Financial health (1-5)": 5, "Geopolitical exposure (1-5)": 1, "Flexibility (1-5)": 5, "Unit cost ($)": 48},
])

# ---------------------------- Helpers ----------------------------
def money(x: float) -> str:
    return f"${x:,.0f}"

def pct(x: float) -> str:
    return f"{x:.1f}%"

def safe_float(x, default=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

def forecast_demand(history: np.ndarray, growth: float, uncertainty: float, method: str) -> np.ndarray:
    """Create a six-month demand forecast from historical data and user assumptions."""
    base = float(np.mean(history[-3:]))
    if method == "Moving average":
        trend = (history[-1] - history[-4]) / 4
    elif method == "Linear trend":
        x = np.arange(len(history))
        slope = np.polyfit(x, history, 1)[0]
        trend = slope
    else:
        trend = 0.0
    forecasts = []
    for i in range(6):
        val = base + trend * (i + 1)
        val *= (1 + growth / 100) ** (i + 1)
        forecasts.append(max(0, val))
    return np.array(forecasts)

def supplier_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Higher score = higher risk; score normalized to 0-100."""
    result = df.copy()
    required_numeric = ["Share (%)", "Lead time (days)", "Quality (%)", "Financial health (1-5)",
                       "Geopolitical exposure (1-5)", "Flexibility (1-5)", "Unit cost ($)"]
    for column in required_numeric:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "Supplier" in result.columns:
        result = result[result["Supplier"].fillna("").astype(str).str.strip() != ""].copy()
    result["Lead time (days)"] = result["Lead time (days)"].fillna(0)
    result["Quality (%)"] = result["Quality (%)"].fillna(95)
    result["Financial health (1-5)"] = result["Financial health (1-5)"].fillna(3)
    result["Geopolitical exposure (1-5)"] = result["Geopolitical exposure (1-5)"].fillna(3)
    result["Flexibility (1-5)"] = result["Flexibility (1-5)"].fillna(3)
    # normalized risk factors
    lead = result["Lead time (days)"].clip(0, 60) / 60 * 25
    quality_risk = (100 - result["Quality (%)"].clip(0, 100)) / 100 * 25
    financial_risk = (5 - result["Financial health (1-5)"].clip(1, 5)) / 4 * 20
    geo = (result["Geopolitical exposure (1-5)"].clip(1, 5) - 1) / 4 * 20
    flexibility = (5 - result["Flexibility (1-5)"].clip(1, 5)) / 4 * 10
    result["Risk score"] = (lead + quality_risk + financial_risk + geo + flexibility).round(1)
    result["Risk category"] = pd.cut(
        result["Risk score"], bins=[-0.1, 30, 60, 100.1],
        labels=["Low", "Medium", "High"]
    ).astype(str)
    return result

def simulate(
    forecast: np.ndarray,
    starting_inventory: float,
    safety_stock: float,
    production_capacity: float,
    production_plan_pct: float,
    supplier_delay_months: int,
    unit_cost: float,
    selling_price: float,
    holding_cost_rate: float,
    stockout_penalty: float,
    fixed_monthly_cost: float,
    initial_cash: float,
    procurement_cost_multiplier: float,
    demand_noise: float = 0.0,
    seed: int = 17,
) -> pd.DataFrame:
    """Deterministic monthly simulation. Demand noise can be enabled for stress-testing."""
    rng = np.random.default_rng(seed)
    inventory = max(0.0, starting_inventory)
    cash = initial_cash
    rows = []
    for i, base_demand in enumerate(forecast):
        demand = max(0.0, base_demand * (1 + rng.normal(0, demand_noise / 100)))
        desired_production = production_capacity * production_plan_pct / 100
        available_production = desired_production if i >= supplier_delay_months else desired_production * 0.75
        production = min(production_capacity, max(0, available_production))
        available = inventory + production
        sales = min(demand, available)
        ending_inventory = max(0.0, available - sales)
        lost_sales = max(0.0, demand - sales)
        revenue = sales * selling_price
        variable_production_cost = production * unit_cost * procurement_cost_multiplier
        holding_cost = ending_inventory * unit_cost * holding_cost_rate / 100
        penalty_cost = lost_sales * stockout_penalty
        total_cost = variable_production_cost + holding_cost + fixed_monthly_cost + penalty_cost
        net_cash_flow = revenue - total_cost
        cash += net_cash_flow
        rows.append({
            "Month": MONTHS[i],
            "Demand": demand,
            "Production": production,
            "Sales": sales,
            "Lost sales": lost_sales,
            "Ending inventory": ending_inventory,
            "Revenue": revenue,
            "Production cost": variable_production_cost,
            "Holding cost": holding_cost,
            "Stockout cost": penalty_cost,
            "Fixed cost": fixed_monthly_cost,
            "Total cost": total_cost,
            "Net cash flow": net_cash_flow,
            "Cash balance": cash,
            "Service level (%)": 100 * sales / demand if demand > 0 else 100,
        })
        inventory = ending_inventory
    return pd.DataFrame(rows)

def create_management_report(
    scenario_name: str,
    sim: pd.DataFrame,
    forecast: np.ndarray,
    supplier_df: pd.DataFrame,
    settings: Dict[str, Any],
    strategy_notes: str,
) -> str:
    total_sales = sim["Sales"].sum()
    total_demand = sim["Demand"].sum()
    service = 100 * total_sales / total_demand if total_demand else 0
    total_revenue = sim["Revenue"].sum()
    total_cost = sim["Total cost"].sum()
    profit = total_revenue - total_cost
    high_risk = supplier_df[supplier_df["Risk category"] == "High"]["Supplier"].tolist()
    return f"""# Board Decision Brief — {scenario_name}

**Prepared:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Decision horizon:** 6 months

## Executive recommendation
Management should align production with the demand outlook, protect service levels through targeted safety stock, diversify high-risk supply exposure, and preserve liquidity. The simulation is directional and should be validated against company data before implementation.

## 1. Demand and production
Forecast demand over six months: **{forecast.sum():,.0f} units**. Planned production rate: **{settings['production_plan_pct']:.0f}% of monthly capacity**; capacity is **{settings['capacity']:,.0f} units/month**. Demand uncertainty assumption: **{settings['uncertainty']:.1f}%**.

## 2. Inventory and procurement
Starting inventory: **{settings['starting_inventory']:,.0f} units**. Safety-stock target: **{settings['safety_stock']:,.0f} units**. Procurement cost multiplier: **{settings['procurement_multiplier']:.2f}x**. Consider supplier allocation and replenishment timing together rather than minimizing unit price alone.

## 3. Supplier risk
Suppliers rated high-risk in the model: **{', '.join(high_risk) if high_risk else 'None'}**. Risk should be reviewed across lead time, quality, financial health, geopolitical exposure, and flexibility.

## 4. Financial performance
Projected revenue: **${total_revenue:,.0f}**; simulated total costs: **${total_cost:,.0f}**; operating contribution before any unmodeled items: **${profit:,.0f}**. Ending cash: **${sim['Cash balance'].iloc[-1]:,.0f}**. Average service level: **{service:.1f}%**.

## 5. Investment / financing
Prioritize investments with measurable payback: supplier qualification, demand sensing, production bottleneck reduction, inventory visibility, and working-capital discipline. Financing decisions should be tested against cash-flow volatility and downside scenarios.

## 6. Board decisions requested
1. Approve the proposed production and inventory policy for a monitored pilot.
2. Authorize a supplier-risk mitigation plan and backup sourcing for critical materials.
3. Set cash, service-level, inventory, and margin guardrails.
4. Require monthly review of forecast error, supplier performance, and scenario assumptions.

## Management notes
{strategy_notes if strategy_notes.strip() else 'No additional management notes supplied.'}

---
*This report is generated from a classroom simulation. It is not financial, legal, or operational advice. Assumptions and model outputs are illustrative.*
"""

def optional_llm_analysis(prompt: str, api_key: str, model: str) -> str:
    """Optional OpenAI integration; returns a clear error if not configured."""
    if not api_key:
        return "No API key supplied. Use the built-in analytical tools and report template, or enter an API key to enable AI analysis."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=model, input=prompt)
        return response.output_text
    except Exception as exc:
        return f"AI call failed: {exc}"

# ---------------------------- Sidebar controls ----------------------------
st.sidebar.title("🏭 Manufacturing Decision Lab")
st.sidebar.caption("Executive team simulator • Six-month planning horizon")
st.sidebar.markdown("### Scenario settings")
scenario = st.sidebar.selectbox("Scenario", ["Base case", "Demand surge", "Demand contraction", "Supplier disruption", "Cost inflation", "Custom"])
scenario_defaults = {
    "Base case": (0.0, 0, 1.00, 10.0),
    "Demand surge": (18.0, 0, 1.05, 18.0),
    "Demand contraction": (-15.0, 0, 0.98, 8.0),
    "Supplier disruption": (3.0, 1, 1.12, 20.0),
    "Cost inflation": (4.0, 0, 1.18, 12.0),
    "Custom": (0.0, 0, 1.00, 10.0),
}
growth_default, delay_default, multiplier_default, noise_default = scenario_defaults[scenario]
growth = st.sidebar.slider("Demand growth assumption (%)", -30.0, 40.0, float(growth_default), 1.0)
supplier_delay = st.sidebar.slider("Supplier disruption delay (months)", 0, 3, int(delay_default))
procurement_multiplier = st.sidebar.slider("Procurement cost multiplier", 0.80, 1.50, float(multiplier_default), 0.01)
uncertainty = st.sidebar.slider("Demand uncertainty / volatility (%)", 0.0, 35.0, float(noise_default), 1.0)
st.sidebar.markdown("### Operating assumptions")
starting_inventory = st.sidebar.number_input("Starting inventory (units)", 0, 100000, 1200, 100)
safety_stock = st.sidebar.number_input("Safety stock target (units)", 0, 100000, 500, 100)
capacity = st.sidebar.number_input("Monthly production capacity (units)", 100, 100000, 1400, 100)
production_plan_pct = st.sidebar.slider("Capacity utilization / planned production (%)", 40, 100, 90, 5)
unit_cost = st.sidebar.number_input("Unit production cost ($)", 1.0, 100000.0, 42.0, 1.0)
selling_price = st.sidebar.number_input("Selling price ($/unit)", 1.0, 100000.0, 95.0, 1.0)
holding_cost_rate = st.sidebar.slider("Monthly holding cost (% of unit cost)", 0.0, 10.0, 1.5, 0.5)
stockout_penalty = st.sidebar.number_input("Lost-sale penalty ($ per unit)", 0.0, 10000.0, 15.0, 1.0)
fixed_monthly_cost = st.sidebar.number_input("Fixed monthly operating cost ($)", 0.0, 10000000.0, 28000.0, 1000.0)
initial_cash = st.sidebar.number_input("Opening cash ($)", 0.0, 100000000.0, 250000.0, 10000.0)

# ---------------------------- Main interface ----------------------------
st.title("🏭 AI Manufacturing Decision Lab")
st.subheader("Capstone: Managing an AI-enabled manufacturing enterprise")
st.markdown(
    "Your group is the executive management team of a manufacturing company facing uncertain demand, "
    "excess inventory, supplier disruption, constrained capacity, rising procurement costs, and deteriorating financial performance. "
    "Use the controls, data, scenario analysis, and optional AI advisor to build a board-ready strategy."
)

with st.expander("📖 Case briefing and objectives", expanded=False):
    st.markdown("""
**Your mandate:** stabilize operations and cash flow, restore profitability, and protect customers over the next six months.

**Required board deliverables**
- Demand and production planning recommendations
- Inventory and procurement optimization
- Supplier risk assessment
- Financial performance analysis
- Investment or financing recommendations
- Executive management report and board presentation

**Suggested group roles:** CEO/strategy lead, COO/operations, CFO, supply-chain director, risk/data lead.
**Timebox:** 10 minutes analysis, 10 minutes strategy, 5 minutes board presentation, 5 minutes peer challenge.
""")

with st.expander("🧭 How to use the simulator", expanded=False):
    st.markdown("""
1. Select a scenario from the sidebar and review assumptions.
2. Adjust demand, capacity, production, inventory, and cost parameters.
3. Examine KPIs and identify trade-offs. A higher production plan may improve service but increase inventory and cash tied up.
4. Explore scenario comparisons and supplier risk.
5. Use the optional AI advisor for structured critique (API key required).
6. Download the management report and simulation data.
""")

# Historical demand and forecast
st.header("1. Demand planning")
history_df = pd.DataFrame({"Month": [f"History {i+1}" for i in range(len(DEFAULT_HISTORY))], "Demand": DEFAULT_HISTORY})
forecast_method = st.radio("Forecast method", ["Moving average", "Linear trend", "Flat baseline"], horizontal=True)
forecast = forecast_demand(DEFAULT_HISTORY, growth, uncertainty, forecast_method)
forecast_df = pd.DataFrame({"Month": MONTHS, "Forecast demand": forecast})
chart_hist = px.line(history_df, x="Month", y="Demand", markers=True, title="Historical demand")
chart_fore = px.line(forecast_df, x="Month", y="Forecast demand", markers=True, title="Six-month demand forecast")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(chart_hist, use_container_width=True)
with c2:
    st.plotly_chart(chart_fore, use_container_width=True)
st.dataframe(forecast_df.round(0), use_container_width=True, hide_index=True)
st.caption("Forecasts are simple classroom models, not statistical demand-planning systems. The volatility setting is used in the simulation.")

# Simulation
st.header("2. Operations and financial simulation")
settings = {
    "production_plan_pct": production_plan_pct,
    "capacity": capacity,
    "starting_inventory": starting_inventory,
    "safety_stock": safety_stock,
    "uncertainty": uncertainty,
    "procurement_multiplier": procurement_multiplier,
}
sim = simulate(
    forecast, starting_inventory, safety_stock, capacity, production_plan_pct,
    supplier_delay, unit_cost, selling_price, holding_cost_rate,
    stockout_penalty, fixed_monthly_cost, initial_cash,
    procurement_multiplier, demand_noise=uncertainty,
)
revenue_total = sim["Revenue"].sum()
cost_total = sim["Total cost"].sum()
profit_total = revenue_total - cost_total
total_demand = sim["Demand"].sum()
total_sales = sim["Sales"].sum()
service_level = 100 * total_sales / total_demand if total_demand else 100
ending_cash = sim["Cash balance"].iloc[-1]
avg_inventory = sim["Ending inventory"].mean()
inventory_turnover_proxy = total_sales / max(avg_inventory, 1)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Projected revenue", money(revenue_total))
k2.metric("Simulated contribution", money(profit_total), help="Revenue minus simulated costs; not full accounting profit.")
k3.metric("Service level", pct(service_level))
k4.metric("Ending cash", money(ending_cash))
k5.metric("Average ending inventory", f"{avg_inventory:,.0f} units")

fig = go.Figure()
fig.add_trace(go.Scatter(x=sim["Month"], y=sim["Demand"], name="Demand", mode="lines+markers"))
fig.add_trace(go.Scatter(x=sim["Month"], y=sim["Production"], name="Production", mode="lines+markers"))
fig.add_trace(go.Scatter(x=sim["Month"], y=sim["Sales"], name="Sales", mode="lines+markers"))
fig.update_layout(title="Demand, production and realized sales", xaxis_title="Planning period", yaxis_title="Units")
st.plotly_chart(fig, use_container_width=True)
fig_inv = px.area(sim, x="Month", y=["Ending inventory", "Lost sales"], title="Inventory and unmet demand")
st.plotly_chart(fig_inv, use_container_width=True)
fig_fin = px.bar(sim, x="Month", y=["Revenue", "Total cost"], barmode="group", title="Revenue versus modeled costs")
st.plotly_chart(fig_fin, use_container_width=True)
st.dataframe(sim.round(2), use_container_width=True, hide_index=True)

with st.expander("🧮 KPI definitions"):
    st.markdown("""
- **Service level:** fulfilled demand / total demand.
- **Ending inventory:** inventory remaining after demand is served.
- **Lost sales:** demand that could not be fulfilled in this simplified model.
- **Simulated contribution:** revenue less production, holding, fixed, and stockout penalty costs.
- **Cash balance:** opening cash plus simulated monthly net cash flows. This simplified cash model assumes revenue and costs settle in the same month.
- **Inventory turnover proxy:** total sales divided by average ending inventory; not a standard annualized turnover ratio.
""")

# Supplier risk module
st.header("3. Supplier risk and sourcing")
st.markdown("Edit the supplier profile to reflect the case your group is assigned. Supplier shares are descriptive; the simulator does not automatically allocate production by share.")
supplier_input = st.data_editor(DEFAULT_SUPPLIERS, num_rows="dynamic", use_container_width=True, key="supplier_editor")
supplier_scored = supplier_risk_score(supplier_input)
supplier_scored["Share (%)"] = pd.to_numeric(supplier_scored["Share (%)"], errors="coerce").fillna(0)
if not supplier_scored.empty:
    risk_chart = px.bar(supplier_scored, x="Supplier", y="Risk score", color="Risk category", range_y=[0, 100], title="Supplier risk score (0 = low risk, 100 = high risk)")
    st.plotly_chart(risk_chart, use_container_width=True)
    st.dataframe(supplier_scored, use_container_width=True, hide_index=True)
    share_total = supplier_scored["Share (%)"].sum()
    if abs(share_total - 100) > 0.1:
        st.warning(f"Supplier allocation shares total {share_total:.1f}%, not 100%. Consider rebalancing.")
    st.caption("Risk score weights: lead time 25%, quality risk 25%, financial health 20%, geopolitical exposure 20%, flexibility 10%. A score is a screening indicator, not a validated risk model.")

# Decision checklist and strategy inputs
st.header("4. Management decisions")
col_a, col_b = st.columns(2)
with col_a:
    inventory_strategy = st.selectbox("Inventory policy", ["Lean / minimize inventory", "Balanced buffer", "Service-first buffer", "Segmented ABC policy"])
    sourcing_strategy = st.selectbox("Sourcing strategy", ["Lowest-cost sourcing", "Dual sourcing", "Supplier diversification", "Strategic partnership / contracts"])
    demand_strategy = st.selectbox("Demand response", ["Conservative forecast", "Consensus planning", "Demand sensing / frequent refresh", "Scenario-based planning"])
with col_b:
    investment = st.selectbox("Investment priority", ["No major investment; conserve cash", "Demand analytics and planning", "Automation / bottleneck capacity", "Supplier qualification and resilience", "Inventory visibility / ERP integration"])
    financing = st.selectbox("Financing recommendation", ["Use internal cash and working-capital release", "Short-term working-capital facility", "Equipment lease / asset financing", "Equity or strategic capital", "Defer financing pending scenario review"])
    strategic_notes = st.text_area("Management rationale / key actions", placeholder="Enter your group's strategy, assumptions, risks, and board asks...", height=150)

decision_summary = {
    "Inventory policy": inventory_strategy,
    "Sourcing strategy": sourcing_strategy,
    "Demand response": demand_strategy,
    "Investment priority": investment,
    "Financing recommendation": financing,
    "Management notes": strategic_notes,
}
st.subheader("Decision summary")
st.json(decision_summary)

# Scenario comparison
st.header("5. Scenario comparison and stress testing")
st.markdown("Compare your current assumptions with a few stress cases. These are simplified, single-run simulations.")
scenario_rows = []
stress_cases = [
    ("Current plan", growth, supplier_delay, procurement_multiplier, uncertainty),
    ("Demand +20%", growth + 20, supplier_delay, procurement_multiplier, uncertainty),
    ("Demand -20%", growth - 20, supplier_delay, procurement_multiplier, uncertainty),
    ("Disruption +1 month", growth, min(3, supplier_delay + 1), procurement_multiplier, uncertainty),
    ("Procurement +15%", growth, supplier_delay, procurement_multiplier * 1.15, uncertainty),
]
for name, g, d, m, u in stress_cases:
    f = forecast_demand(DEFAULT_HISTORY, g, u, forecast_method)
    s = simulate(f, starting_inventory, safety_stock, capacity, production_plan_pct, d, unit_cost,
                 selling_price, holding_cost_rate, stockout_penalty, fixed_monthly_cost, initial_cash,
                 m, demand_noise=u, seed=17)
    rev = s["Revenue"].sum()
    cost = s["Total cost"].sum()
    scenario_rows.append({
        "Scenario": name,
        "Revenue ($)": rev,
        "Modeled costs ($)": cost,
        "Contribution ($)": rev - cost,
        "Service level (%)": 100 * s["Sales"].sum() / s["Demand"].sum() if s["Demand"].sum() else 100,
        "Ending cash ($)": s["Cash balance"].iloc[-1],
        "Ending inventory (units)": s["Ending inventory"].iloc[-1],
    })
scenario_df = pd.DataFrame(scenario_rows)
st.dataframe(scenario_df.round(1), use_container_width=True, hide_index=True)
scenario_chart = px.bar(scenario_df, x="Scenario", y="Contribution ($)", title="Contribution across stress cases")
st.plotly_chart(scenario_chart, use_container_width=True)

# Risk dashboard
st.header("6. Risk register")
risk_items = [
    ("Demand forecast error", "Medium", "High", "Refresh forecast weekly; monitor forecast bias and error."),
    ("Single-source dependence", "High", "High", "Qualify alternate supplier; negotiate allocation and contingency clauses."),
    ("Inventory obsolescence", "Medium", "Medium", "Segment inventory; use lifecycle and aging reports; pause slow movers."),
    ("Capacity bottleneck", "High", "High", "Identify constraint; prioritize product mix; test overtime or targeted capex."),
    ("Cash-flow squeeze", "High", "High", "Cash forecast; working-capital release; financing trigger and covenant review."),
    ("Supplier quality", "Medium", "High", "Incoming inspection, supplier corrective action, quality scorecards."),
    ("Cyber / AI governance", "Medium", "Medium", "Protect data; validate AI outputs; assign accountable decision owner."),
]
risk_df = pd.DataFrame(risk_items, columns=["Risk", "Likelihood", "Impact", "Mitigation"])
st.dataframe(risk_df, use_container_width=True, hide_index=True)

# Optional AI Advisor
st.header("7. Generative AI management advisor")
st.info("AI is optional. This module can produce a structured critique of your strategy when an OpenAI API key is configured. Avoid entering confidential company or personal data.")
api_key = st.text_input("OpenAI API key (optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
ai_model = st.selectbox("AI model", ["gpt-5-mini", "gpt-5", "gpt-4.1-mini"], index=0)
if st.button("Generate AI boardroom critique", type="primary"):
    prompt = f"""Act as a skeptical but constructive manufacturing-company board advisor for an MBA group case.
Use only the data and decisions supplied below; distinguish assumptions from facts, do not invent external benchmarks.
Give a concise but substantive critique with headings: (1) Executive verdict, (2) Demand and capacity, (3) Inventory and procurement, (4) Supplier resilience, (5) Financial implications and liquidity, (6) Investment/financing, (7) Three prioritized actions, (8) Board questions.
Simulation summary:
Scenario: {scenario}
Forecast total demand: {forecast.sum():.0f}
Simulated demand: {sim['Demand'].sum():.0f}
Sales: {sim['Sales'].sum():.0f}
Lost sales: {sim['Lost sales'].sum():.0f}
Revenue: ${revenue_total:.2f}
Modeled costs: ${cost_total:.2f}
Contribution: ${profit_total:.2f}
Service level: {service_level:.2f}%
Ending cash: ${ending_cash:.2f}
Ending inventory: {sim['Ending inventory'].iloc[-1]:.0f}
Supplier risk table: {supplier_scored.to_dict(orient='records')}
Management choices: {json.dumps(decision_summary, ensure_ascii=False)}
"""
    with st.spinner("AI advisor is reviewing the strategy..."):
        ai_text = optional_llm_analysis(prompt, api_key, ai_model)
    st.session_state["ai_board_critique"] = ai_text
if "ai_board_critique" in st.session_state:
    st.markdown(st.session_state["ai_board_critique"])

# Report and exports
st.header("8. Executive report and downloads")
report = create_management_report(scenario, sim, forecast, supplier_scored, settings, strategic_notes)
report += "\n\n## Selected management decisions\n"
for key, value in decision_summary.items():
    report += f"\n**{key}:** {value if value else 'Not specified'}\n"
report += "\n## Scenario comparison\n\n" + scenario_df.to_markdown(index=False)
if "ai_board_critique" in st.session_state:
    report += "\n\n## AI boardroom critique\n\n" + st.session_state["ai_board_critique"]
st.download_button("⬇️ Download board report (Markdown)", report, file_name="manufacturing_board_report.md", mime="text/markdown")
st.download_button("⬇️ Download simulation data (CSV)", sim.to_csv(index=False), file_name="manufacturing_simulation.csv", mime="text/csv")
st.download_button("⬇️ Download scenario comparison (CSV)", scenario_df.to_csv(index=False), file_name="scenario_comparison.csv", mime="text/csv")
st.download_button("⬇️ Download supplier risk data (CSV)", supplier_scored.to_csv(index=False), file_name="supplier_risk.csv", mime="text/csv")

# Boardroom pitch scaffold
st.header("9. Board presentation guide")
st.markdown("""
Use this structure for a **5-minute board presentation**:
1. **Situation:** What is happening to demand, inventory, supply, capacity, and cash?
2. **Diagnosis:** Which two or three issues drive the deterioration?
3. **Decision:** What will management change immediately, and what will it defer?
4. **Trade-offs:** What service, margin, risk, and cash trade-offs are accepted?
5. **Financial case:** What is the expected impact and what assumptions drive it?
6. **Risk and contingency:** What could invalidate the plan and what is the trigger for action?
7. **Board ask:** What approvals, resources, or limits are requested?

**Peer challenge:** another group should ask one question about the forecast, one about cash, and one about supplier resilience.
""")
st.caption("Classroom simulation • Outputs are illustrative and depend on student-selected assumptions.")
