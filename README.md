# Rossmann Store Sales Forecasting

End-to-end time series forecasting pipeline for 1,115 Rossmann drugstores across Germany — combining feature engineering, ensemble modeling, and SHAP-based interpretability to surface actionable business insights.

---

## Preview

![Prediction vs Actual](assets/prediction_vs_actual.png)

![SHAP Summary](assets/shap_summary.png)

![QQ Plot](assets/qq_plot.png)

---

# 📊 Key Results

| Model | RMSPE | Notes |
|---|---|---|
| Ridge Regression | ~0.164 | Linear baseline |
| Facebook Prophet | 0.29 – 1.8+ | Per-store, high variance |
| XGBoost | 0.143 | Best — store-level normalization + log transform |

✅ **12.8% improvement over Ridge baseline**

---

# 🗂️ Repository Structure

```text
rossmann-forecasting/
├── TimeSeriAnalysis.ipynb
├── train.csv
├── test.csv
├── store.csv
├── df_merged.parquet
└── README.md
```

---

# 🔄 Pipeline Overview

```text
Raw CSV (train + store)
        ↓
Data Cleaning & Merging
- Filter closed days (Open = 0)
- Memory optimization (dtype casting)
- Export → df_merged.parquet

        ↓

Feature Engineering
- Time features (cyclical encoding)
- Lag features (lag_7, lag_14, lag_28, lag_365)
- Rolling statistics (mean/std, 7–56 days)
- Holiday proximity
- Promo interaction features

        ↓

Modeling
- Ridge Regression
- Facebook Prophet
- XGBoost

        ↓

Interpretability & Evaluation
- SHAP Summary Plot
- Residual Analysis + QQ Plot
- Interactive Plotly visualization
```

---

# ⚙️ Feature Engineering

## Time Features

| Feature | Description |
|---|---|
| DayOfWeek_sin/cos | Cyclical encoding for weekdays |
| Month_sin/cos | Cyclical encoding for months |
| IsWeekend | Weekend indicator |
| IsMonthStart / End | Month boundary flags |
| DaysBeforeHoliday | Distance to nearest holiday |

---

## Lag & Rolling Features

| Feature | Description |
|---|---|
| lag_7 / 14 / 28 / 365 | Historical sales intervals |
| rolling_mean_7/28/56 | Trend signals |
| rolling_std_7/28/56 | Volatility signals |
| Sales_ExpandingMean | Store-level cumulative average |

---

## Interaction Features

| Feature | Description |
|---|---|
| promo_month | Promotion × month interaction |
| promo_weekend | Promotion × weekend interaction |
| store_sales_mean | Per-store normalization |
| competition_near | Nearby competitor flag |

---

# 🤖 Modeling

## Ridge Regression — Baseline

Linear model with L2 regularization.  
Serves as a stable baseline and sanity check for feature validity.

- RMSPE: ~0.164
- Stable train/validation gap

---

## Facebook Prophet — Trend Decomposition

Applied per-store for interpretability rather than raw accuracy.

Captures:
- Trend
- Weekly seasonality
- Promotion effects

However, performance variance across stores is high:

- RMSPE: 0.29 – 1.8+

---

## XGBoost — Main Model

```python
XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)
```

### Store-level Target Normalization

```python
df["sales_norm"] = df["Sales"] / df["store_sales_mean"]
df["sales_norm_log"] = np.log1p(df["sales_norm"])
```

Normalizing by each store's average sales before log-transform reduces inter-store variance and stabilizes the target distribution.

Result:

- RMSPE improved from **0.149 → 0.143**

---

# 🔍 SHAP Interpretability

![SHAP Summary](assets/shap_summary.png)

## Business Insights

### 1. Promo — The #1 Sales Driver

Promotions show the strongest positive SHAP impact across the dataset.

Days with active promotions consistently push predictions upward, making promotion timing the highest-ROI controllable variable.

---

### 2. DayOfWeek — Hidden Complexity

High weekday values (Fri/Sat/Sun) appear on both positive and negative SHAP ranges.

This reveals operational differences between stores, especially Sunday closures.

---

### 3. Short-term Momentum

Features such as:
- rolling_mean_7
- lag_7
- lag_14

show strong predictive power for near-term forecasting.

---

### 4. Promo × Time Interaction

Month-end promotions generate amplified positive effects.

This suggests promotion scheduling near month boundaries increases impact.

---

### 5. Low-signal Features

Features with minimal SHAP contribution:
- StoreType
- SchoolHoliday
- CompetitionDistance

The model largely ignores these directly.

---

# 📉 Residual Analysis

![QQ Plot](assets/qq_plot.png)

## QQ Plot Findings

### Mid-range Quantiles

Predictions closely follow the diagonal line.

→ Strong performance on normal trading days.

---

### Right Tail

The model under-predicts extreme sales spikes.

Likely causes:
- Log-transform smoothing
- Missing event-specific features

---

### Left Tail

Minor over-prediction on very slow days.

Magnitude is much smaller than right-tail error.

---

## Key Implication

XGBoost performs strongly on typical business conditions but struggles with Black Friday-style spikes.

Potential solution:
- Spike detector
- Hybrid ensemble model

---

# 🏁 Model Comparison Summary

| Dimension | Ridge | Prophet | XGBoost |
|---|---|---|---|
| Forecast Accuracy | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Interpretability | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Stability | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Use Case | Baseline | Behavior Analysis | Production Forecasting |

---

# 🚀 How to Run

## 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost prophet shap plotly matplotlib seaborn pyarrow statsmodels
```

---

## 2. Download Dataset

Download dataset from:

- Kaggle — Rossmann Store Sales

Place:
- `train.csv`
- `test.csv`
- `store.csv`

inside the project root.

---

## 3. Run Notebook

```bash
jupyter lab TimeSeriAnalysis.ipynb
```

Run all cells sequentially.

The notebook automatically generates:

```text
df_merged.parquet
```

during preprocessing.

---

# 📌 Screenshots

```text
shap_summary.png
qq_plot.png
prediction_vs_actual.png
feature_importance.png
```

---

# 🔮 Future Improvements

- Spike modeling using external event calendars
- LightGBM / CatBoost benchmarking
- Store clustering before modeling
- Probabilistic forecasting with confidence intervals

---

# 📚 Dataset

**Rossmann Store Sales — Kaggle Competition**

- 1,017,209 training records
- 1,115 stores
- January 2013 – July 2015
