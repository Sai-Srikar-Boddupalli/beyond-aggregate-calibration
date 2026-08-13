import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb

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
df['stratify_key'] = df['is_default'].astype(str) + "_" + df['income_bracket']

pos_ratio = (len(df) - df['is_default'].sum()) / df['is_default'].sum()
defaulters = df[df['is_default'] == 1].copy()

print("\n" + "="*70)
print("--- TEST 1: RAW INT_RATE DISTRIBUTION (TRUE DEFAULTERS) ---")
int_rate_stats = defaulters.groupby('income_bracket')['int_rate'].agg(['mean', 'median']).reset_index()
int_rate_stats['income_bracket'] = pd.Categorical(int_rate_stats['income_bracket'], categories=['High', 'Middle', 'Low'], ordered=True)
print(int_rate_stats.sort_values('income_bracket').to_string(index=False))
print("="*70)

# --- PREP FOR CV ---
categorical_cols = ["purpose", "home_ownership", "term"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = df['is_default']

# ==========================================
# TEST 2: BLIND MODEL (No annual_inc)
# ==========================================
print("\nBuilding and running BLIND model (no annual_inc) for int_rate decile check...")
numeric_zeros_blind = [
    "loan_amnt", "int_rate", "dti", "fico_range_low", "revol_util", 
    "pub_rec_bankruptcies", "pub_rec_bankruptcies_missing", "revol_util_missing"
]
numeric_neg_one = ["emp_length", "emp_length_missing"]

preprocessor_blind = ColumnTransformer(transformers=[
    ("num_zeros", SimpleImputer(strategy="constant", fill_value=0), numeric_zeros_blind),
    ("num_neg_ones", SimpleImputer(strategy="constant", fill_value=-1), numeric_neg_one),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
])

blind_pipeline = Pipeline([
    ("preprocess", preprocessor_blind),
    ("classifier", xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1))
])

X_blind = df.drop(columns=['is_default', 'loan_status', 'stratify_key', 'income_bracket', 'annual_inc'])
df['cv_pred_prob_blind'] = cross_val_predict(blind_pipeline, X_blind, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]

print("\n--- TEST 2: TPR CONTROLLED BY INT_RATE DECILE ---")
defaulters['cv_pred_class_blind'] = (df.loc[defaulters.index, 'cv_pred_prob_blind'] > 0.5).astype(int)
defaulters['int_rate_decile'] = pd.qcut(defaulters['int_rate'], 10, labels=False, duplicates='drop')

int_rate_tpr = defaulters.groupby(['int_rate_decile', 'income_bracket']).agg(
    total=('is_default', 'count'),
    caught=('cv_pred_class_blind', 'sum')
).reset_index()

int_rate_tpr['tpr'] = int_rate_tpr['caught'] / int_rate_tpr['total']
pivot_tpr = int_rate_tpr.pivot(index='int_rate_decile', columns='income_bracket', values='tpr')
pivot_tpr = pivot_tpr[['High', 'Middle', 'Low']]

print("TPR by Interest Rate Decile within True Defaulters (0 = Lowest Rates, 9 = Highest Rates):")
print(pivot_tpr.map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "NaN"))
print("="*70)

# ==========================================
# TEST 3: DOUBLE-BLIND MODEL (No annual_inc, No int_rate)
# ==========================================
print("\nBuilding and running DOUBLE-BLIND model (no annual_inc, no int_rate)...")
numeric_zeros_db = [
    "loan_amnt", "dti", "fico_range_low", "revol_util", 
    "pub_rec_bankruptcies", "pub_rec_bankruptcies_missing", "revol_util_missing"
]

preprocessor_db = ColumnTransformer(transformers=[
    ("num_zeros", SimpleImputer(strategy="constant", fill_value=0), numeric_zeros_db),
    ("num_neg_ones", SimpleImputer(strategy="constant", fill_value=-1), numeric_neg_one),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
])

db_pipeline = Pipeline([
    ("preprocess", preprocessor_db),
    ("classifier", xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1))
])

X_db = df.drop(columns=['is_default', 'loan_status', 'stratify_key', 'income_bracket', 'annual_inc', 'int_rate'])
df['cv_pred_prob_db'] = cross_val_predict(db_pipeline, X_db, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]

print("\n--- TEST 3: EQUAL OPPORTUNITY TPR (DOUBLE-BLIND MODEL) ---")
defaulters['cv_pred_class_db'] = (df.loc[defaulters.index, 'cv_pred_prob_db'] > 0.5).astype(int)

db_tpr_results = defaulters.groupby('income_bracket').agg(
    total_actual_defaults=('is_default', 'count'),
    correctly_predicted_defaults=('cv_pred_class_db', 'sum')
).reset_index()

db_tpr_results['recall_tpr'] = (db_tpr_results['correctly_predicted_defaults'] / db_tpr_results['total_actual_defaults']) * 100
db_tpr_results['income_bracket'] = pd.Categorical(db_tpr_results['income_bracket'], categories=['High', 'Middle', 'Low'], ordered=True)

print("Residual gap after removing BOTH income and upstream interest rate assignments:")
print(db_tpr_results.sort_values('income_bracket').to_string(index=False, float_format="%.2f%%"))
print("="*70 + "\n")

from sklearn.metrics import roc_auc_score, accuracy_score

print("\n--- TEST 4: DOUBLE-BLIND MODEL DEGRADATION CHECK ---")
# Baseline model metrics (using the standard threshold for accuracy)
y_true = df['is_default']
db_prob = df['cv_pred_prob_db']
db_class = (db_prob > 0.5).astype(int)

auc = roc_auc_score(y_true, db_prob)
acc = accuracy_score(y_true, db_class)

print(f"Double-Blind CV AUC Score: {auc:.4f}")
print(f"Double-Blind CV Accuracy:  {acc:.4f}")
print("="*70 + "\n")