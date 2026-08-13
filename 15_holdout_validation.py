import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

print("Loading data for strict hold-out validation...")
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

print("Performing strict 80/20 temporal/random split...")
# Drop target and features we are blinding
X = df.drop(columns=['is_default', 'loan_status', 'income_bracket', 'annual_inc', 'int_rate'])
y = df['is_default']
brackets = df['income_bracket']

# Split data - keeping track of the income brackets for evaluation
X_train, X_test, y_train, y_test, bracket_train, bracket_test = train_test_split(
    X, y, brackets, test_size=0.20, random_state=42, stratify=df['is_default'].astype(str) + "_" + df['income_bracket']
)

pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()

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

db_pipeline = Pipeline([
    ("preprocess", preprocessor_db),
    ("classifier", xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1))
])

print("Training Double-Blind XGBoost solely on 80% train set...")
db_pipeline.fit(X_train, y_train)

print("Predicting on fully unseen 20% test set...")
test_pred_probs = db_pipeline.predict_proba(X_test)[:, 1]

# Reconstruct the test set for evaluation
test_results = pd.DataFrame({
    'is_default': y_test,
    'income_bracket': bracket_test,
    'pred_prob': test_pred_probs
})
test_results['pred_class'] = (test_results['pred_prob'] > 0.5).astype(int)

# Isolate actual defaulters in the test set
test_defaulters = test_results[test_results['is_default'] == 1].copy()

holdout_tpr = test_defaulters.groupby('income_bracket').agg(
    total_actual_defaults=('is_default', 'count'),
    correctly_predicted_defaults=('pred_class', 'sum')
).reset_index()

holdout_tpr['recall_tpr'] = (holdout_tpr['correctly_predicted_defaults'] / holdout_tpr['total_actual_defaults']) * 100
holdout_tpr['income_bracket'] = pd.Categorical(holdout_tpr['income_bracket'], categories=['High', 'Middle', 'Low'], ordered=True)

print("\n" + "="*70)
print("--- TRUE HOLDOUT (20%) TPR: DOUBLE-BLIND XGBOOST ---")
print(holdout_tpr.sort_values('income_bracket').to_string(index=False, float_format="%.2f%%"))
print("="*70 + "\n")