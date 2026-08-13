import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import shap

print("Loading data...")
cols = [
    'loan_amnt', 'annual_inc', 'int_rate', 'dti', 
    'fico_range_low', 'revol_util', 'pub_rec_bankruptcies', 'emp_length',
    'purpose', 'home_ownership', 'term', 'loan_status'
]

df = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols=cols, low_memory=False)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['is_default'] = (df['loan_status'] == 'Charged Off').astype(int)

# --- Missingness ---
for col in ["pub_rec_bankruptcies", "revol_util"]:
    df[f"{col}_missing"] = df[col].isna().astype(int)
df['emp_length'] = df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float)
df['emp_length_missing'] = df['emp_length'].isna().astype(int)

# --- Post-Hoc Labels ---
conditions = [
    (df['annual_inc'] < 50000),
    (df['annual_inc'] >= 50000) & (df['annual_inc'] < 100000),
    (df['annual_inc'] >= 100000)
]
df['income_bracket'] = np.select(conditions, ['Low', 'Middle', 'High'], default='Unknown')
df = df[df['income_bracket'] != 'Unknown'].copy()

pos_ratio = (len(df) - df['is_default'].sum()) / df['is_default'].sum()

print("Fitting the BLIND model (no annual_inc)...")
categorical_cols = ["purpose", "home_ownership", "term"]
numeric_zeros = [
    "loan_amnt", "int_rate", "dti", "fico_range_low", "revol_util", 
    "pub_rec_bankruptcies", "pub_rec_bankruptcies_missing", "revol_util_missing"
]
numeric_neg_one = ["emp_length", "emp_length_missing"]

preprocessor = ColumnTransformer(transformers=[
    ("num_zeros", SimpleImputer(strategy="constant", fill_value=0), numeric_zeros),
    ("num_neg_ones", SimpleImputer(strategy="constant", fill_value=-1), numeric_neg_one),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
])

X_raw = df.drop(columns=['is_default', 'loan_status', 'income_bracket', 'annual_inc'])
y = df['is_default']

X_transformed = preprocessor.fit_transform(X_raw)
feature_names = preprocessor.get_feature_names_out()

model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1)
model.fit(X_transformed, y)

print("Identifying High-Income False Negatives...")
# Predict probabilities on the whole set to find the misses
pred_probs = model.predict_proba(X_transformed)[:, 1]
df['pred_class'] = (pred_probs > 0.5).astype(int)

# Mask for High-Income False Negatives: True label is 1, Predicted is 0, Bracket is High
fn_mask = (df['is_default'] == 1) & (df['pred_class'] == 0) & (df['income_bracket'] == 'High')
X_high_fn = X_transformed[fn_mask]

print(f"Total High-Income False Negatives isolated: {len(X_high_fn)}")
print("Calculating SHAP values for this specific subgroup...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_high_fn)

# Calculate the mean SIGNED impact
signed_shap_df = pd.DataFrame(shap_values, columns=feature_names)
mean_signed_impact = signed_shap_df.mean().sort_values()

print("\n" + "="*70)
print("--- SIGNED SHAP IMPACT (High-Income False Negatives) ---")
print("Negative values mean the feature pushed the model TOWARD non-default (safe).")
print("\nTop 10 features suppressing the risk score:")
print(mean_signed_impact.head(10).round(4).to_string())
print("="*70 + "\n")