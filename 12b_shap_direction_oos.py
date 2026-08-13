import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
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

print("Performing strict 80/20 split for Out-of-Sample SHAP...")
X = df.drop(columns=['is_default', 'loan_status', 'income_bracket', 'annual_inc'])
y = df['is_default']
brackets = df['income_bracket']

# Stratified split to ensure proper default/bracket representation
X_train, X_test, y_train, y_test, bracket_train, bracket_test = train_test_split(
    X, y, brackets, test_size=0.20, random_state=42, stratify=df['is_default'].astype(str) + "_" + df['income_bracket']
)

pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()

print("Fitting the BLIND model on 80% train set...")
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

blind_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1))
])

blind_pipeline.fit(X_train, y_train)

print("Predicting and isolating High-Income False Negatives on 20% UNSEEN TEST SET...")
# Transform X_test manually to feed into SHAP later
X_test_transformed = blind_pipeline.named_steps['preprocess'].transform(X_test)
feature_names = blind_pipeline.named_steps['preprocess'].get_feature_names_out()

# Predict on test set
test_pred_probs = blind_pipeline.named_steps['classifier'].predict_proba(X_test_transformed)[:, 1]
test_pred_class = (test_pred_probs > 0.5).astype(int)

# Mask for High-Income FNs purely in the test set
fn_mask_test = (y_test == 1) & (test_pred_class == 0) & (bracket_test == 'High')
X_high_fn_test = X_test_transformed[fn_mask_test]

print(f"Total out-of-sample High-Income False Negatives isolated: {len(X_high_fn_test)}")
print("Calculating SHAP values...")

explainer = shap.TreeExplainer(blind_pipeline.named_steps['classifier'])
shap_values_test = explainer.shap_values(X_high_fn_test)

signed_shap_df_test = pd.DataFrame(shap_values_test, columns=feature_names)
mean_signed_impact_test = signed_shap_df_test.mean().sort_values()

print("\n" + "="*70)
print("--- OUT-OF-SAMPLE SIGNED SHAP IMPACT (High-Income False Negatives) ---")
print("Negative values mean the feature pushed the model TOWARD non-default (safe).")
print("\nTop 10 features suppressing the risk score on unseen data:")
print(mean_signed_impact_test.head(10).round(4).to_string())
print("="*70 + "\n")