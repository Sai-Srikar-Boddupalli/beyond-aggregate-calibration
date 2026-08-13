import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb
import shap

print("Loading data for double-blind monotonicity check...")
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

print("\nBuilding DOUBLE-BLIND model (no annual_inc, no int_rate)...")
categorical_cols = ["purpose", "home_ownership", "term"]
numeric_zeros_db = [
    "loan_amnt", "dti", "fico_range_low", "revol_util", 
    "pub_rec_bankruptcies", "pub_rec_bankruptcies_missing", "revol_util_missing"
]
numeric_neg_one = ["emp_length", "emp_length_missing"]

preprocessor_db = ColumnTransformer(transformers=[
    ("num_zeros", SimpleImputer(strategy="constant", fill_value=0), numeric_zeros_db),
    ("num_neg_ones", SimpleImputer(strategy="constant", fill_value=-1), numeric_neg_one),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
])

X_raw_db = df.drop(columns=['is_default', 'loan_status', 'income_bracket', 'annual_inc', 'int_rate'])
y = df['is_default']

# Fit preprocessor and model for SHAP
X_transformed_db = preprocessor_db.fit_transform(X_raw_db)
feature_names_db = preprocessor_db.get_feature_names_out()

model_db = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1)
model_db.fit(X_transformed_db, y)

print("Calculating SHAP values for Middle vs Low False Negatives...")
pred_probs = model_db.predict_proba(X_transformed_db)[:, 1]
df['pred_class_db'] = (pred_probs > 0.5).astype(int)

# Isolate Middle and Low FNs
fn_middle_mask = (df['is_default'] == 1) & (df['pred_class_db'] == 0) & (df['income_bracket'] == 'Middle')
fn_low_mask = (df['is_default'] == 1) & (df['pred_class_db'] == 0) & (df['income_bracket'] == 'Low')

X_fn_middle = X_transformed_db[fn_middle_mask]
X_fn_low = X_transformed_db[fn_low_mask]

explainer = shap.TreeExplainer(model_db)
shap_middle = explainer.shap_values(X_fn_middle)
shap_low = explainer.shap_values(X_fn_low)

mean_shap_middle = pd.DataFrame(shap_middle, columns=feature_names_db).mean()
mean_shap_low = pd.DataFrame(shap_low, columns=feature_names_db).mean()

comparison_df = pd.DataFrame({
    'Middle_FN_Mean_SHAP': mean_shap_middle,
    'Low_FN_Mean_SHAP': mean_shap_low
})
comparison_df['Diff_Absolute'] = (comparison_df['Middle_FN_Mean_SHAP'] - comparison_df['Low_FN_Mean_SHAP']).abs()

print("\n" + "="*70)
print("--- TEST 1: SIGNED SHAP COMPARISON (Middle vs Low False Negatives) ---")
print(comparison_df.sort_values('Diff_Absolute', ascending=False).head(10).round(4).to_string())
print("="*70)

print("\nRunning Cross-Validation for Decile Check... (This will take a minute)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['cv_pred_prob_db'] = cross_val_predict(
    Pipeline([("preprocess", preprocessor_db), ("classifier", xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1))]),
    X_raw_db, y, cv=cv, method='predict_proba', n_jobs=1
)[:, 1]

print("\n--- TEST 2: TPR CONTROLLED BY LOAN_AMNT DECILE (DOUBLE-BLIND) ---")
defaulters = df[df['is_default'] == 1].copy()
defaulters['cv_pred_class_db'] = (defaulters['cv_pred_prob_db'] > 0.5).astype(int)
defaulters['loan_amnt_decile'] = pd.qcut(defaulters['loan_amnt'], 10, labels=False, duplicates='drop')

loan_tpr = defaulters.groupby(['loan_amnt_decile', 'income_bracket']).agg(
    total=('is_default', 'count'),
    caught=('cv_pred_class_db', 'sum')
).reset_index()

loan_tpr['tpr'] = loan_tpr['caught'] / loan_tpr['total']
pivot_loan = loan_tpr.pivot(index='loan_amnt_decile', columns='income_bracket', values='tpr')
pivot_loan = pivot_loan[['High', 'Middle', 'Low']]

print("TPR by Loan Amount Decile within True Defaulters (0 = Smallest Loans, 9 = Largest Loans):")
print(pivot_loan.map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "NaN"))
print("="*70 + "\n")