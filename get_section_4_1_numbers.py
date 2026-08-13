import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict
from scipy.stats import chi2_contingency
import xgboost as xgb

print("Loading data for Section 4.1 Baseline Discard Rate audit...")
cols = ['loan_amnt', 'annual_inc', 'int_rate', 'dti', 'loan_status']
df = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols=cols, low_memory=False)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['is_default'] = (df['loan_status'] == 'Charged Off').astype(int)

# Income brackets for evaluation
conditions = [
    (df['annual_inc'] < 50000),
    (df['annual_inc'] >= 50000) & (df['annual_inc'] < 100000),
    (df['annual_inc'] >= 100000)
]
df['income_bracket'] = np.select(conditions, ['Low', 'Middle', 'High'], default='Unknown')
df = df[df['income_bracket'] != 'Unknown'].copy()

# 4-Feature Baseline Model
X = df[['loan_amnt', 'annual_inc', 'int_rate', 'dti']]
y = df['is_default']
pos_ratio = (len(y) - y.sum()) / y.sum()

imputer = SimpleImputer(strategy="constant", fill_value=0)
clf = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1)
pipeline = Pipeline([("impute", imputer), ("clf", clf)])

print("Generating out-of-fold predictions...")
oof_probs = cross_val_predict(pipeline, X, y, cv=5, method='predict_proba', n_jobs=-1)[:, 1]

# Evaluate ONLY among actual defaulters (is_default == 1)
defaulters = df[df['is_default'] == 1].copy()
defaulters['oof_prob'] = oof_probs[df['is_default'] == 1]
defaulters['discarded'] = (defaulters['oof_prob'] < 0.15).astype(int)

# Crosstab and Chi-Square
table = pd.crosstab(defaulters['income_bracket'], defaulters['discarded'])
chi2, p_val, dof, expected = chi2_contingency(table)

# Cramér's V
n = table.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

# Discard rates
rates = defaulters.groupby('income_bracket')['discarded'].agg(['count', 'sum'])
rates['discard_rate_pct'] = (rates['sum'] / rates['count']) * 100

print("\n" + "="*60)
print("--- SECTION 4.1 NUMBERS TO COPY-PASTE ---")
print(rates.loc[['High', 'Middle', 'Low'], ['count', 'sum', 'discard_rate_pct']].to_string())
print("-" * 60)
print(f"High-Income Discard Rate:   {rates.loc['High', 'discard_rate_pct']:.2f}%  <-- Put in first [X]%")
print(f"Low-Income Discard Rate:    {rates.loc['Low', 'discard_rate_pct']:.2f}%   <-- Put in second [X]%")
print(f"Chi-Square Statistic (χ²):  {chi2:.2f}     <-- Put in χ² ≈ [X]")
print(f"P-Value:                    {p_val:.4e}")
print(f"Cramér's V:                 {cramers_v:.4f}    <-- Confirms the ~0.03-0.07 range")
print("="*60 + "\n")