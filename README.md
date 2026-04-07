📈 Rossmann Store Sales Forecasting (Time Series Analysis)

📌 Overview
This project applies Time Series Analysis and Machine Learning to forecast daily sales for the Rossmann drugstore chain. It covers an end-to-end data pipeline, including Exploratory Data Analysis (EDA), Feature Engineering, predictive modeling, and model interpretability.

📁 Repository Structure
TimeSeriAnalysis.ipynb: The main Jupyter Notebook containing all the code for data processing, EDA, modeling, and evaluation.

train.csv, test.csv, store.csv: Raw datasets (Kaggle).

df_merged.parquet: The cleaned, merged, and memory-optimized dataset.

README.md: Project documentation.

⚙️ Workflow
Data Preprocessing & EDA: Handled missing values, filtered out closed days (Open = 0), and applied log1p transformation to the target variable (Sales) to stabilize variance.

Feature Engineering: Extracted time-based features (cyclical encoding), lag features (lag_7, lag_14, lag_365), rolling statistics (mean/std), and external factors (Promos, Holidays).

Modeling: * Ridge Regression: Used as a baseline model.

Prophet: Applied for univariate trend and seasonality analysis on representative stores.

XGBoost: The main predictive model.

Model Interpretability: Utilized SHAP (SHapley Additive exPlanations) to explain the XGBoost model's predictions and extract business insights.

🏆 Key Results
Performance: The XGBoost model (with target normalization and log transform) achieved the best performance with an RMSPE of ~0.143 on the validation set, improving the baseline by 12.8%.

Business Insights: SHAP analysis revealed that Promo is the strongest sales driver. Additionally, purchasing behavior is heavily influenced by the day of the week (DayOfWeek) and short-term historical momentum (lag and rolling features).

🚀 How to Run
1. Install dependencies:

Bash
pip install pandas numpy scikit-learn xgboost prophet shap matplotlib seaborn pyarrow
2. Execute the notebook:
Open TimeSeriAnalysis.ipynb in Jupyter Notebook / Jupyter Lab and run all cells sequentially.
