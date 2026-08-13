import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

print("Loading expanded dataset...")
# Adding the expanded features
cols = [
    'loan_amnt', 'annual_inc', 'int_rate', 'dti', 
    'fico_range_low', 'revol_util', 'pub_rec_bankruptcies', 'emp_length',
    'purpose', 'home_ownership', 'term', 'loan_status'
]

# We use read_csv_auto equivalent in pandas (treating mixed types safely)
df = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols=cols, low_memory=False)

# Filter to finished loans only
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['is_default'] = (df['loan_status'] == 'Charged Off').astype(int)

# --- Feature Engineering: Row-wise operations (No Leakage Risk) ---
print("Applying row-wise missingness flags...")
missing_flag_cols = ["pub_rec_bankruptcies", "revol_util", "emp_length"]
for col in missing_flag_cols:
    df[f"{col}_missing"] = df[col].isna().astype(int)

# Clean up emp_length (convert "10+ years" to 10, etc.)
df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)

# Create Income Brackets for Stratification
conditions = [
    (df['annual_inc'] < 50000),
    (df['annual_inc'] >= 50000) & (df['annual_inc'] < 100000),
    (df['annual_inc'] >= 100000)
]
df['income_bracket'] = np.select(conditions, ['Low', 'Middle', 'High'], default='Unknown')

# Create a combined stratification key (Target + Demographic)
df['stratify_key'] = df['is_default'].astype(str) + "_" + df['income_bracket']

# Drop records where income is completely missing to keep the audit clean
df = df[df['income_bracket'] != 'Unknown'].copy()

# --- Pipeline Architecture ---
print("Building the strict preprocessor pipeline...")
categorical_cols = ["purpose", "home_ownership", "term"]
numeric_zeros = ["loan_amnt", "annual_inc", "int_rate", "dti", "fico_range_low", "revol_util", "pub_rec_bankruptcies"] + [f"{c}_missing" for c in missing_flag_cols]
numeric_neg_one = ["emp_length"]

preprocessor = ColumnTransformer(transformers=[
    ("num_zeros", SimpleImputer(strategy="constant", fill_value=0), numeric_zeros),
    ("num_neg_ones", SimpleImputer(strategy="constant", fill_value=-1), numeric_neg_one),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", xgb.XGBClassifier(eval_metric='logloss', n_jobs=-1))
])

# --- Stratified K-Fold Execution ---
print("Executing Stratified 5-Fold Cross-Validation... (This will take a few minutes)")
X = df.drop(columns=['is_default', 'loan_status', 'stratify_key', 'income_bracket'])
y = df['is_default']
strat_key = df['stratify_key']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Generate out-of-fold predictions
df['cv_pred_prob'] = cross_val_predict(model_pipeline, X, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]

# Calculate overall baseline metrics
df['cv_pred_class'] = (df['cv_pred_prob'] > 0.5).astype(int)
print("\n--- ROBUST BASELINE RESULTS ---")
print(f"CV Accuracy: {accuracy_score(y, df['cv_pred_class']):.4f}")
print(f"CV AUC Score: {roc_auc_score(y, df['cv_pred_prob']):.4f}")

# Re-run the specific noise audit logic
df['is_discarded'] = ((df['is_default'] == 1) & (df['cv_pred_prob'] < 0.15))

audit_results = df.groupby('income_bracket').agg(
    total_applicants=('is_default', 'count'),
    discarded_records=('is_discarded', 'sum')
).reset_index()

audit_results['percent_discarded'] = (audit_results['discarded_records'] / audit_results['total_applicants']) * 100
print("\n--- FAIRNESS AUDIT: INCOME BIAS (ROBUST MODEL) ---")
print(audit_results.sort_values('percent_discarded', ascending=False).to_string(index=False))
print("\n--- CONFOUND CHECK: OLS REGRESSION ---")
print("Fitting OLS Regression to isolate income's effect on predicted probability...")

# Ensure income_bracket is treated as a category, with 'Low' as the reference baseline
defaulters['income_bracket'] = pd.Categorical(
    defaulters['income_bracket'], 
    categories=['Low', 'Middle', 'High'], 
    ordered=True
)

# Run the regression on the continuous probability score
ols_model = smf.ols(formula="cv_pred_prob ~ C(income_bracket) + loan_amnt + dti + fico_range_low + emp_length", data=defaulters)
results = ols_model.fit()

print("\n--- REGRESSION RESULTS (TRUE DEFAULTERS ONLY) ---")
print(results.summary().tables[1])