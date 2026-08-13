import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

print("--- ITEM 7: COERCION SILENT-FAILURE CHECK ---")
df_raw = pd.read_csv("accepted_2007_to_2018Q4.csv", nrows=100000, low_memory=False) # Sample for speed
check_cols = ['emp_length', 'dti', 'annual_inc', 'loan_amnt', 'int_rate', 'revol_util']

for col in check_cols:
    if col in df_raw.columns:
        original_nulls = df_raw[col].isna().sum()
        coerced_series = pd.to_numeric(df_raw[col].astype(str).str.extract(r'(\d+\.?\d*)')[0] if col == 'emp_length' else df_raw[col], errors='coerce')
        new_nulls = coerced_series.isna().sum()
        print(f"{col}: {original_nulls} original NaNs -> {new_nulls} NaNs after coercion (Lost {new_nulls - original_nulls} to formatting)")

print("\n--- ITEM 6: EXPANDED CORRELATION MATRIX ---")
cols = ['annual_inc', 'fico_range_low', 'dti', 'int_rate', 'loan_amnt', 'loan_status']
df = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols=cols, low_memory=False)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()

# Fast clean for correlation
for c in cols[:-1]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

corr_matrix = df[['annual_inc', 'fico_range_low', 'dti', 'int_rate', 'loan_amnt']].corr(method='spearman')
print("Spearman Rank Correlation (Robust to outliers):")
print(corr_matrix.round(3).to_string())

print("\n--- ITEM 4: FORMAL STATISTICAL TESTS (Holdout Set Results) ---")
# Hardcoding the exact counts from your Script 15 Holdout run to generate formal test stats
counts = np.array([5277, 11149])    # High-income caught, Low-income caught
nobs = np.array([8814, 17859])      # High-income total defaults, Low-income total defaults

z_stat, p_val = proportions_ztest(counts, nobs, alternative='two-sided')
(ci_low_high, ci_low_low), (ci_up_high, ci_up_low) = proportion_confint(counts, nobs, alpha=0.05, method='normal')

tpr_high = counts[0] / nobs[0]
tpr_low = counts[1] / nobs[1]
gap = tpr_low - tpr_high
ci_gap_lower = gap - 1.96 * np.sqrt((tpr_high * (1 - tpr_high) / nobs[0]) + (tpr_low * (1 - tpr_low) / nobs[1]))
ci_gap_upper = gap + 1.96 * np.sqrt((tpr_high * (1 - tpr_high) / nobs[0]) + (tpr_low * (1 - tpr_low) / nobs[1]))

print(f"Double-Blind Holdout TPR Gap (Low vs High): {gap*100:.2f}%")
print(f"Z-Statistic: {z_stat:.4f}")
print(f"P-Value: {p_val:.4e}")
print(f"95% CI for the difference: [{ci_gap_lower*100:.2f}%, {ci_gap_upper*100:.2f}%]")