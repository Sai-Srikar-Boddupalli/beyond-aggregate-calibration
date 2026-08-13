import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

print("Loading data for Strict Out-of-Sample SHAP Analysis...")
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

print("Performing strict 80/20 split...")
X = df.drop(columns=['is_default', 'loan_status', 'income_bracket'])
y = df['is_default']
brackets = df['income_bracket']

X_train, X_test, y_train, y_test, bracket_train, bracket_test = train_test_split(
    X, y, brackets, test_size=0.20, random_state=42, stratify=df['is_default'].astype(str) + "_" + df['income_bracket']
)

# Fix for #3: Compute pos_ratio strictly on train set
pos_ratio_train = (len(y_train) - y_train.sum()) / y_train.sum()

# --- PREPROCESSORS ---
categorical_cols = ["purpose", "home_ownership", "term"]
numeric_neg_one = ["emp_length", "emp_length_missing"]

# Blind Preprocessor (Keeps int_rate)
numeric_zeros_blind = ["loan_amnt", "int_rate", "dti", "fico_range_low", "revol_util", "pub_rec_bankruptcies", "pub_rec_bankruptcies_missing", "revol_util_missing"]
preprocessor_blind = ColumnTransformer([
    ("num_zeros", SimpleImputer(strategy="constant", fill_value=0), numeric_zeros_blind),
    ("num_neg_ones", SimpleImputer(strategy="constant", fill_value=-1), numeric_neg_one),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
])

# Double-Blind Preprocessor (Drops int_rate)
numeric_zeros_db = ["loan_amnt", "dti", "fico_range_low", "revol_util", "pub_rec_bankruptcies", "pub_rec_bankruptcies_missing", "revol_util_missing"]
preprocessor_db = ColumnTransformer([
    ("num_zeros", SimpleImputer(strategy="constant", fill_value=0), numeric_zeros_db),
    ("num_neg_ones", SimpleImputer(strategy="constant", fill_value=-1), numeric_neg_one),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
])

# --- 1. BLIND MODEL SHAP (High-Income Direction Check) ---
print("\nFitting BLIND Model on Train Set...")
X_train_blind = X_train.drop(columns=['annual_inc'])
X_test_blind = X_test.drop(columns=['annual_inc'])

pipe_blind = Pipeline([
    ("prep", preprocessor_blind),
    ("clf", xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio_train))
])
pipe_blind.fit(X_train_blind, y_train)

X_test_blind_tf = pipe_blind.named_steps['prep'].transform(X_test_blind)
feat_names_blind = pipe_blind.named_steps['prep'].get_feature_names_out()

test_pred_blind = (pipe_blind.predict_proba(X_test_blind)[:, 1] > 0.5).astype(int)
fn_high_mask = (y_test == 1) & (test_pred_blind == 0) & (bracket_test == 'High')
X_fn_high_test = X_test_blind_tf[fn_high_mask]

print(f"High-Income False Negatives isolated (Test Set Only): {len(X_fn_high_test)}")
explainer_blind = shap.TreeExplainer(pipe_blind.named_steps['clf'])
shap_high = explainer_blind.shap_values(X_fn_high_test)

print("\n--- OOS SIGNED SHAP: HIGH-INCOME FNs (Blind Model) ---")
print(pd.DataFrame(shap_high, columns=feat_names_blind).mean().sort_values().head(10).round(4).to_string())

# --- 2. DOUBLE-BLIND MODEL SHAP (Middle vs Low Comparison) ---
print("\nFitting DOUBLE-BLIND Model on Train Set...")
X_train_db = X_train.drop(columns=['annual_inc', 'int_rate'])
X_test_db = X_test.drop(columns=['annual_inc', 'int_rate'])

pipe_db = Pipeline([
    ("prep", preprocessor_db),
    ("clf", xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio_train))
])
pipe_db.fit(X_train_db, y_train)

X_test_db_tf = pipe_db.named_steps['prep'].transform(X_test_db)
feat_names_db = pipe_db.named_steps['prep'].get_feature_names_out()

test_pred_db = (pipe_db.predict_proba(X_test_db)[:, 1] > 0.5).astype(int)
fn_mid_mask = (y_test == 1) & (test_pred_db == 0) & (bracket_test == 'Middle')
fn_low_mask = (y_test == 1) & (test_pred_db == 0) & (bracket_test == 'Low')

X_fn_mid_test = X_test_db_tf[fn_mid_mask]
X_fn_low_test = X_test_db_tf[fn_low_mask]

print(f"Middle-Income FNs isolated (Test Only): {len(X_fn_mid_test)}")
print(f"Low-Income FNs isolated (Test Only): {len(X_fn_low_test)}")

explainer_db = shap.TreeExplainer(pipe_db.named_steps['clf'])
shap_mid = explainer_db.shap_values(X_fn_mid_test)
shap_low = explainer_db.shap_values(X_fn_low_test)

mean_shap_mid = pd.DataFrame(shap_mid, columns=feat_names_db).mean()
mean_shap_low = pd.DataFrame(shap_low, columns=feat_names_db).mean()

comp_df = pd.DataFrame({'Middle_FN_Mean': mean_shap_mid, 'Low_FN_Mean': mean_shap_low})
comp_df['Diff_Absolute'] = (comp_df['Middle_FN_Mean'] - comp_df['Low_FN_Mean']).abs()

print("\n--- OOS SIGNED SHAP: MIDDLE vs LOW FNs (Double-Blind Model) ---")
print(comp_df.sort_values('Diff_Absolute', ascending=False).head(10).round(4).to_string())
print("="*70 + "\n")