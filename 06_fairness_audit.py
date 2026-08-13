import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_predict

print("Loading data for the Fairness Audit...")
df = pd.read_csv("baseline_data.csv", usecols=['loan_amnt', 'annual_inc', 'int_rate', 'dti', 'is_default']).dropna()

X = df[['loan_amnt', 'annual_inc', 'int_rate', 'dti']]
y = df['is_default']

print("Re-running the noise detection to capture the discarded records...")
model = xgb.XGBClassifier(eval_metric='logloss')
df['default_prob'] = cross_val_predict(model, X, y, cv=3, method='predict_proba')[:, 1]

# Recreate our exact definition of "Noise"
df['is_discarded'] = ((df['is_default'] == 1) & (df['default_prob'] < 0.15))

print("\n--- FAIRNESS AUDIT: INCOME BIAS ---")
# Create Income Brackets to see who is getting penalized by the cleaning algorithm
conditions = [
    (df['annual_inc'] < 50000),
    (df['annual_inc'] >= 50000) & (df['annual_inc'] < 100000),
    (df['annual_inc'] >= 100000)
]
choices = ['Low (<$50k)', 'Middle ($50k-$100k)', 'High (>$100k)']
df['income_bracket'] = np.select(conditions, choices, default='Unknown')

# Calculate the deletion rate for each group
audit_results = df.groupby('income_bracket').agg(
    total_applicants=('is_default', 'count'),
    discarded_records=('is_discarded', 'sum')
).reset_index()

# Calculate the percentage of records deleted in each bracket
audit_results['percent_discarded'] = (audit_results['discarded_records'] / audit_results['total_applicants']) * 100

# Sort it neatly
audit_results = audit_results.sort_values('percent_discarded', ascending=False)
print(audit_results.to_string(index=False))