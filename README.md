# 📡 Mobile Network Capacity Planning & Saturation Risk Forecasting

## Executive Summary
Mobile network operators must continuously anticipate future network load to avoid congestion, optimize CAPEX allocation, and support commercial growth.

This project implements an **end-to-end capacity planning pipeline** that:
- forecasts mobile network load at short and mid-term horizons,
- estimates **probabilistic saturation risk** at cell and regional levels,
- evaluates **what-if demand growth scenarios** (e.g. +20% subscribers),
- delivers **decision-ready reports** for engineering and business stakeholders.

The outputs are directly usable to **prioritize network upgrades and investments**.

---

## Business Context
Operators must decide:
- where to deploy new base stations,
- when to upgrade technologies (4G → 5G / 5G SA),
- how to prevent congestion during peak hours.

Inaccurate forecasts can result in **millions of euros of misallocated CAPEX** or degraded Quality of Service.

This project focuses on **capacity risk anticipation**, not only point forecasts.

---

## Key Features
- **Multi-step traffic forecasting** (J+7 / J+30)
- **Window-based saturation probability**
- **Zone-specific capacity thresholds** (urban / suburban / rural)
- **Scenario analysis** (commercial demand shock)
- **Interactive and executive-ready reports** (HTML & PDF)

---

## Data Overview
Synthetic but realistic multi-year hourly data per cell:
- Network traffic (Mbps)
- Active users
- Region
- Zone type: `urban`, `suburban`, `rural`
- Time features (hour, weekday, month, seasonality)

### Capacity Thresholds (per zone)
| Zone type | Saturation threshold |
|---------|----------------------|
| Urban | 800 Mbps |
| Suburban | 450 Mbps |
| Rural | 250 Mbps |

The pipeline is designed to be **directly connectable to real operator data**.

---

## Modeling Approach

### Traffic Forecasting
- Gradient Boosted Trees (XGBoost)
- Auto-regressive time-series features
- Multi-step forecasting:
  - **J+7 (168 hours)**
  - **J+30 (720 hours)**

### Saturation Risk Definition
Instead of evaluating single-point forecasts, saturation risk is defined as:

> **Probability that network load exceeds capacity at least once over the forecast horizon**

Formally:
P( max(traffic_t over horizon) > capacity_threshold )

---


### Uncertainty Modeling
- Residual bootstrap simulation
- Robust calibration:
  - winsorization (quantile clipping)
  - sigma clipping
- Window-based probabilistic risk

### Risk Levels
| Probability | Risk level |
|-----------|-----------|
| p < 0.25 | LOW |
| 0.25 ≤ p < 0.60 | MEDIUM |
| p ≥ 0.60 | HIGH |

---

## What-If Scenarios
The project includes **commercial growth simulations**:
- **+20% subscribers**

Demand growth is modeled as a first-order traffic scaling proxy, enabling fast assessment of **capacity stress under growth scenarios**.

This allows clear answers to questions such as:
> “What happens to network capacity if subscriber adoption accelerates?”

---

## Outputs

### Data Reports
- `capacity_risk_horizons.csv`
  Baseline risk at J+7 and J+30
- `capacity_risk_whatif_users_1p2.csv`
  Risk under +20% subscriber scenario

### Interactive Visual Reports (HTML)
- Baseline saturation risk by region
- Risk increase under demand growth
- Baseline vs what-if comparison heatmaps

### Executive Reports (PDF)
- Baseline saturation risk by region
- What-if risk delta by region
- Baseline vs what-if comparative heatmap

These reports are **ready to be shared with technical and executive teams**.

---

## How to Run

### Environment Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run Forecasting & Risk Pipeline
python -m src.ncf.run_mvp

## Generate Interactive Visual Reports
python -m src.ncf.generate_reports

## Generate Executive PDF Reports
python -m src.ncf.export_pdf

> Note: PDF export relies on Plotly + Kaleido.
> If needed, install Chrome for Kaleido:
plotly_get_chrome

---

## Business Value

This project demonstrates the ability to:

address core mobile network capacity challenges,

translate ML forecasts into operational risk indicators,

evaluate CAPEX impact of demand growth scenarios,

deliver outputs usable by engineering, product, and strategy teams.

---

## Future Enhancements

Joint modeling of users and traffic demand

Quantile or conformal prediction

Spatial correlation between neighboring cells

Cost-based CAPEX optimization framework

---

## Author

## Daniel MONCADA LEON
Data Scientist — Applied Machine Learning & Network Analytics

---

## ✅ Status

Production-ready analytical prototype
Suitable for mobile network capacity planning use cases.
