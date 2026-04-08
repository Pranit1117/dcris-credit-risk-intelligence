# ⬡ DCRIS — Dynamic Credit Risk Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**A production-grade, three-layer credit risk intelligence platform built for Indian banks (HDFC · ICICI · SBI)**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Dashboard](#-dashboard) · [Models](#-models)

</div>

---

## 🎯 Overview

DCRIS goes far beyond traditional loan default prediction. It delivers a **three-layer intelligence framework** that:

1. **Predicts** probability of default (ensemble of LR + RF + XGBoost)
2. **Forecasts when** a customer will default using survival analysis (Weibull AFT)
3. **Segments** customers into Safe / Risky / Critical clusters (KMeans)
4. **Explains** every prediction with SHAP values
5. **Recommends** actionable interventions for relationship managers

---

## ✨ Features

| Layer | Method | Output |
|-------|--------|--------|
| Default Prediction | Logistic Regression, Random Forest, XGBoost (ensemble) | Default probability (0–100%) |
| Time-to-Default | Weibull AFT Survival Model | Estimated months until default |
| Risk Segmentation | KMeans Clustering (k=3) | Safe / Risky / Critical label |
| Explainability | SHAP TreeExplainer | Per-feature impact, beeswarm, waterfall |
| Dashboard | Streamlit (dark fintech UI) | 4 interactive pages |

---

## 🏗 Architecture

```
dcris/
├── run.py                    # Entry point (train + launch)
├── requirements.txt
├── config/
│   └── config.py             # Paths, thresholds, feature list
├── src/
│   ├── data_generator.py     # Synthetic Indian banking dataset (15k rows)
│   ├── preprocessor.py       # Missing values, outliers, feature engineering
│   ├── trainer.py            # 3-layer model training pipeline
│   ├── predictor.py          # Single + batch inference
│   ├── explainer.py          # SHAP global & individual explanations
│   └── logger.py             # Structured logging
├── app/
│   ├── main.py               # Streamlit entry point + global CSS
│   ├── data_loader.py        # Cached data & model loader
│   └── pages/
│       ├── home.py           # KPI dashboard + portfolio overview
│       ├── predict.py        # Individual customer prediction
│       ├── analytics.py      # Clusters, feature importance, patterns
│       └── explainability.py # SHAP global + per-customer waterfall
├── models/                   # Serialised .pkl model files
├── data/                     # Raw & processed CSV datasets
└── logs/                     # dcris.log (structured timestamped logs)
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-org/dcris.git
cd dcris
pip install -r requirements.txt
```

### 2. Train models + launch dashboard

```bash
python run.py
```

This will:
- Generate the 15,000-row synthetic dataset
- Run the full preprocessing pipeline
- Train all 5 models (LR, RF, XGB, Weibull AFT, KMeans)
- Launch the Streamlit dashboard at `http://localhost:8501`

### 3. Other modes

```bash
python run.py --train   # train only, no app
python run.py --app     # app only (skip training if models exist)
```

---

## 📊 Dataset

15,000 synthetic customers modelled on Indian banking demographics:

| Feature | Description |
|---------|-------------|
| `income` | Monthly income (₹15k–₹5L, log-normal) |
| `credit_score` | CIBIL-style score (300–900) |
| `loan_amount` | Total outstanding loan (₹10k–₹50L) |
| `emi` | Monthly EMI obligation |
| `credit_utilization_ratio` | % of credit limit used |
| `missed_payments` | Lifetime missed payment count |
| `payment_behavior_score` | Composite score (0–100) |
| `balance_trend` | Monthly balance change (%) |
| `rolling_missed_3m` | Missed payments in last 3 months |
| `banking_relationship_length` | Years with bank |
| `time_to_default` | Survival duration in months |

Default rate: ~25–30% (realistic for MSME + retail segments)

---

## 🤖 Models

### Layer 1 — Default Prediction

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline, `class_weight=balanced` |
| Random Forest | 200 trees, depth 10, balanced |
| XGBoost | 300 estimators, `scale_pos_weight` for imbalance |

**Ensemble**: 50% XGBoost + 30% RF + 20% LR

### Layer 2 — Time-to-Default

**Weibull AFT (Accelerated Failure Time)** via `lifelines`:
- Event: `defaulted = 1`
- Duration: `time_to_default` (months, censored at 36)
- Outputs median survival time for each customer

### Layer 3 — Risk Segmentation

**KMeans (k=3)** on scaled feature matrix. Clusters are post-labelled by mean default probability:
- Lowest mean prob → **Safe**
- Medium → **Risky**
- Highest → **Critical**

---

## 🌐 Dashboard Pages

| Page | Contents |
|------|----------|
| 🏠 Home | Portfolio KPIs, donut chart, scatter, box plots, top-10 risk table |
| 🔍 Predict | Customer form → gauge chart, survival curve, model ensemble breakdown, action |
| 📊 Analytics | 3D scatter, radar chart, heatmap, behavioural distributions |
| 🧠 Explainability | Global SHAP bar + beeswarm, per-customer waterfall |

---

## 📋 Recommended Bank Actions

| Risk Level | Trigger | Action |
|-----------|---------|--------|
| ✅ Safe | Prob < 30% | No action; eligible for credit limit increase |
| ⚠️ Risky | 30–60% | EMI restructuring, credit counselling, monthly monitoring |
| 🚨 Critical | > 60% | NPA prevention protocol, OTS offer, risk committee escalation |

---

## 🔧 Configuration

Edit `config/config.py` to adjust:

```python
RISK_THRESHOLDS = {"safe": 0.3, "risky": 0.6, "critical": 1.0}
N_CLUSTERS = 3
TEST_SIZE = 0.2
```

---

## 📈 Model Monitoring (Roadmap)

- [ ] PSI (Population Stability Index) drift detection
- [ ] Monthly AUC tracking via scheduled retraining
- [ ] Prometheus + Grafana metrics endpoint
- [ ] Webhook alerts when portfolio critical-risk % exceeds threshold

---

## 📄 License

MIT — free to use, modify, and deploy.

---

<div align="center">
Built with ❤️ for Indian banking · DCRIS v2.1
</div>
