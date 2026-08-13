import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_predict

print("Loading data... (This is getting exciting)")
columns_to_use = ['loan_amnt', 'annual_inc', 'int_rate', 'dti', 'is_default']
df = pd.read_csv("baseline_data.csv", usecols=columns_to_use).dropna()

X = df[['loan_amnt', 'annual_inc', 'int_rate', 'dti']]
y = df['is_default']

print("Hunting for label noise using cross-validation... (This takes a minute or two)")
model = xgb.XGBClassifier(eval_metric='logloss')

# cross_val_predict trains the model in blind chunks so it never cheats
df['default_probability'] = cross_val_predict(model, X, y, cv=3, method='predict_proba')[:, 1]

# Querying for the extreme contradictions
print("Filtering out the anomalies...")
# Data says Paid (0) but model is > 85% sure they Defaulted
suspicious_paid = df[(df['is_default'] == 0) & (df['default_probability'] > 0.85)]

# Data says Defaulted (1) but model is < 15% sure they Defaulted (meaning it thought they were safe)
suspicious_default = df[(df['is_default'] == 1) & (df['default_probability'] < 0.15)]

print(f"\n--- NOISE DETECTED ---")
print(f"Suspicious 'Fully Paid' records: {len(suspicious_paid):,}")
print(f"Suspicious 'Charged Off' records: {len(suspicious_default):,}")

# Save these weird records to look at them closer
suspicious_paid.to_csv("noise_suspicious_paid.csv", index=False)
suspicious_default.to_csv("noise_suspicious_default.csv", index=False)
print("Saved the suspicious records to new CSV files!")