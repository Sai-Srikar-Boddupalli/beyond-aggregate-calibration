import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

print("Loading data for Logistic Regression robustness check...")
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
df['stratify_key'] = df['is_default'].astype(str) + "_" + df['income_bracket']

print("Building Double-Blind Logistic Regression Pipeline...")
categorical_cols = ["purpose", "home_ownership", "term"]
numeric_zeros_db = [
    "loan_amnt", "dti", "fico_range_low", "revol_util", 
    "pub_rec_bankruptcies", "pub_rec_bankruptcies_missing", "revol_util_missing"
]
numeric_neg_one = ["emp_length", "emp_length_missing"]

# Linear models require scaling
numeric_transformer_zeros = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", StandardScaler())
])
numeric_transformer_neg_ones = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    ("scaler", StandardScaler())
])

preprocessor_db = ColumnTransformer(transformers=[
    ("num_zeros", numeric_transformer_zeros, numeric_zeros_db),
    ("num_neg_ones", numeric_transformer_neg_ones, numeric_neg_one),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
])

# Logistic Regression with balanced class weights
lr_pipeline = Pipeline([
    ("preprocess", preprocessor_db),
    ("classifier", LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
])

X_db = df.drop(columns=['is_default', 'loan_status', 'stratify_key', 'income_bracket', 'annual_inc', 'int_rate'])
y = df['is_default']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running Cross-Validation... (This will take a minute)")
df['cv_pred_prob_lr'] = cross_val_predict(lr_pipeline, X_db, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

print("\n" + "="*70)
print("--- LOGISTIC REGRESSION: DOUBLE-BLIND TPR CHECK ---")
defaulters = df[df['is_default'] == 1].copy()
defaulters['cv_pred_class_lr'] = (defaulters['cv_pred_prob_lr'] > 0.5).astype(int)

lr_tpr_results = defaulters.groupby('income_bracket').agg(
    total_actual_defaults=('is_default', 'count'),
    correctly_predicted_defaults=('cv_pred_class_lr', 'sum')
).reset_index()

lr_tpr_results['recall_tpr'] = (lr_tpr_results['correctly_predicted_defaults'] / lr_tpr_results['total_actual_defaults']) * 100
lr_tpr_results['income_bracket'] = pd.Categorical(lr_tpr_results['income_bracket'], categories=['High', 'Middle', 'Low'], ordered=True)

print(lr_tpr_results.sort_values('income_bracket').to_string(index=False, float_format="%.2f%%"))
print("="*70 + "\n")