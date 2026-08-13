import pandas as pd
import numpy as np

print("Loading data for profile check...")
cols = [
    'loan_amnt', 'annual_inc', 'dti', 
    'fico_range_low', 'revol_util', 'loan_status'
]

df = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols=cols, low_memory=False)
df = df[df['loan_status'] == 'Charged Off'].copy() # Only true defaulters

# --- Post-Hoc Labels ---
conditions = [
    (df['annual_inc'] < 50000),
    (df['annual_inc'] >= 50000) & (df['annual_inc'] < 100000),
    (df['annual_inc'] >= 100000)
]
df['income_bracket'] = np.select(conditions, ['Low', 'Middle', 'High'], default='Unknown')
df = df[df['income_bracket'] != 'Unknown'].copy()

print("\n" + "="*70)
print("--- DEFAULTER FINANCIAL PROFILE BY INCOME BRACKET ---")

# Calculate medians for robust comparison against outliers (like extreme DTI anomalies)
profile_stats = df.groupby('income_bracket')[['dti', 'revol_util', 'fico_range_low', 'loan_amnt']].median().reset_index()
profile_stats['income_bracket'] = pd.Categorical(profile_stats['income_bracket'], categories=['High', 'Middle', 'Low'], ordered=True)

print("MEDIAN VALUES FOR TRUE DEFAULTERS:")
print(profile_stats.sort_values('income_bracket').to_string(index=False))
print("="*70 + "\n")