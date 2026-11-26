# AI Manufacturing Decision Lab — Streamlit Capstone

A comprehensive interactive classroom simulator for the capstone **“Managing an AI-enabled manufacturing enterprise.”** Student groups act as the executive management team of a hypothetical manufacturer experiencing demand uncertainty, excess inventory, supplier disruptions, constrained capacity, procurement inflation, and deteriorating financial performance.

## What students can do

- Explore a six-month demand forecast using moving-average, linear-trend, or flat-baseline approaches.
- Change demand-growth and uncertainty assumptions.
- Configure operating assumptions: inventory, safety stock, production capacity/utilization, unit cost, selling price, holding costs, stockout penalties, fixed operating costs, cash, procurement-cost multiplier, and supplier-delay scenarios.
- Simulate demand, production, fulfilled sales, lost sales, ending inventory, costs, revenue, cash flow, and service level.
- Compare base and stress scenarios (demand increase/decrease, supplier delay, procurement inflation).
- Edit a supplier profile and evaluate a transparent supplier-risk score across lead time, quality, financial health, geopolitical exposure, and flexibility.
- Record management choices for inventory, sourcing, demand planning, investment, and financing.
- Generate a board-report draft and download report/data tables as Markdown or CSV.
- Optionally ask an OpenAI model to critique the group's strategy and board recommendations.

The app is intentionally usable **without an API key**. All simulations, dashboards, scenario comparisons, and the templated management report run locally.

## Files

- `app.py` — Streamlit application and simulation logic.
- `requirements.txt` — Python package dependencies.

## Setup

Recommended Python: **3.10+**.

### 1. Create a virtual environment

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

## Optional: enable the generative AI advisor

The AI advisor is optional. It uses the OpenAI Python SDK and requires an API key.

### Option A — Enter key in the app

Paste your API key into the AI advisor field in the sidebar/main app. Do not share the key or include it in submitted reports.

### Option B — Environment variable

Windows PowerShell (for the current session):

```powershell
$env:OPENAI_API_KEY="your-key"
streamlit run app.py
```

macOS / Linux:

```bash
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

The model selector in the app defaults to `gpt-5-mini`. Confirm model availability for your account. API usage may incur charges.

**Privacy note:** Do not put confidential company data, personally identifiable information, credentials, or sensitive commercial details into the AI prompt. Student-generated case data is fictional.

## Classroom case brief

### Situation

The company is facing:
- uncertain demand,
- excess inventory,
- supplier disruptions,
- limited production capacity,
- increasing procurement costs,
- worsening financial performance.

### Group task

Each group acts as the executive management team and prepares a recommendation to the board.

### Required deliverables

1. Demand and production planning recommendation.
2. Inventory and procurement optimization plan.
3. Supplier-risk assessment and resilience actions.
4. Financial performance analysis.
5. Investment or financing recommendation.
6. Executive report and board presentation.

### Suggested facilitation

A 30–40 minute capstone format:

| Activity | Suggested time |
|---|---:|
| Read brief and assign group roles | 3–5 min |
| Analyze the scenario and dashboard | 8–10 min |
| Build decisions and stress-test | 8–10 min |
| Prepare board pitch | 5 min |
| Present and peer challenge | 5–10 min |

Suggested roles: CEO/strategy lead, CFO, COO, supply-chain lead, risk/data lead. In smaller groups, combine roles.

## Model structure and limitations

This is a teaching model, not a full ERP, MRP, forecasting, optimization, or financial planning system.

- Forecast methods are intentionally simple and transparent.
- The simulation uses a simplified monthly production/inventory balance.
- Supplier delay is represented as a production reduction for the selected initial months.
- Supplier allocation percentages are descriptive; they do not automatically control the production simulation.
- Cash flow assumes same-period cash receipt and expense settlement, with no receivables/payables timing, taxes, debt service, capex, depreciation, or detailed working-capital mechanics.
- Contribution is revenue less the modeled costs. It should not be interpreted as audited operating profit or EBITDA.
- The risk score is a heuristic, not a validated supplier-risk rating.
- Results are sensitive to student inputs and assumptions.

Students should explicitly state assumptions, uncertainty, limitations, and implementation metrics in their board presentation. They should not treat AI-generated content as a substitute for managerial judgment.

## Suggested board-report structure

1. Executive verdict and recommendation.
2. Demand forecast and capacity implications.
3. Inventory and procurement decisions.
4. Supplier risk and resilience.
5. Financial analysis and liquidity.
6. Investment / financing.
7. Risks, contingencies, trigger points.
8. Decisions and approvals requested from the board.

## Suggested evaluation rubric (100 points)

| Criterion | Points |
|---|---:|
| Demand forecast and production logic | 15 |
| Inventory/procurement decisions | 15 |
| Supplier risk and mitigation | 15 |
| Financial reasoning and cash-flow awareness | 20 |
| Investment / financing logic | 15 |
| Use and critique of AI (verification, assumptions, responsible use) | 10 |
| Board communication and defensible trade-offs | 10 |
| **Total** | **100** |

## Troubleshooting

- **`streamlit` command not found:** ensure the virtual environment is activated and `pip install -r requirements.txt` completed.
- **OpenAI import/API errors:** install requirements again; if AI is not required, the rest of the app still works. Check key, model availability, and billing if an API call fails.
- **Port already in use:** run `streamlit run app.py --server.port 8502`.

## License

This repository is made publicly available for educational, academic, and classroom use. Permission is granted to access and use the code for learning, teaching, and academic demonstration purposes. Redistribution, commercial use, modification for public distribution, or incorporation into other projects requires prior written permission from the copyright holder, unless otherwise specified. This repository is provided "as is", without warranty of any kind. The author assumes no responsibility for any loss or damage arising from its use. For other permissions or licensing inquiries, please contact the author.
