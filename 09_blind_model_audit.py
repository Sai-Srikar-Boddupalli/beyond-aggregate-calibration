import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb

print("Loading data and creating post-hoc labels...")
cols = [
    'loan_amnt', 'annual_inc', 'int_rate', 'dti', 
    'fico_range_low', 'revol_util', 'pub_rec_bankruptcies', 'emp_length',
    'purpose', 'home_ownership', 'term', 'loan_status'
]

df = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols=cols, low_memory=False)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['is_default'] = (df['loan_status'] == 'Charged Off').astype(int)

# --- Missingness Flags ---
for col in ["pub_rec_bankruptcies", "revol_util"]:
    df[f"{col}_missing"] = df[col].isna().astype(int)
df['emp_length'] = df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float)
df['emp_length_missing'] = df['emp_length'].isna().astype(int)

# --- Post-Hoc Grouping (Not for Training) ---
conditions = [
    (df['annual_inc'] < 50000),
    (df['annual_inc'] >= 50000) & (df['annual_inc'] < 100000),
    (df['annual_inc'] >= 100000)
]
df['income_bracket'] = np.select(conditions, ['Low', 'Middle', 'High'], default='Unknown')
df = df[df['income_bracket'] != 'Unknown'].copy()
df['stratify_key'] = df['is_default'].astype(str) + "_" + df['income_bracket']

pos_ratio = (len(df) - df['is_default'].sum()) / df['is_default'].sum()

print("\nBuilding BLIND pipeline (annual_inc strictly removed from training)...")

categorical_cols = ["purpose", "home_ownership", "term"]
# CRITICAL: annual_inc is removed from numeric_zeros
numeric_zeros = [
    "loan_amnt", "int_rate", "dti", "fico_range_low", "revol_util", 
    "pub_rec_bankruptcies", "pub_rec_bankruptcies_missing", "revol_util_missing"
]
numeric_neg_one = ["emp_length", "emp_length_missing"]

preprocessor = ColumnTransformer(transformers=[
    ("num_zeros", SimpleImputer(strategy="constant", fill_value=0), numeric_zeros),
    ("num_neg_ones", SimpleImputer(strategy="constant", fill_value=-1), numeric_neg_one),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1))
])

# CRITICAL: Drop annual_inc from X so the model is truly blind to income
X = df.drop(columns=['is_default', 'loan_status', 'stratify_key', 'income_bracket', 'annual_inc'])
y = df['is_default']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running Cross-Validation on BLIND model (This will take a few minutes)...")
df['cv_pred_prob'] = cross_val_predict(model_pipeline, X, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]

print("\n" + "="*50)
defaulters = df[df['is_default'] == 1].copy()
defaulters['cv_pred_class'] = (defaulters['cv_pred_prob'] > 0.5).astype(int)

print("--- CHECK 1: MEAN/MEDIAN PROBABILITY (TRUE DEFAULTERS) ---")
prob_stats = defaulters.groupby('income_bracket')['cv_pred_prob'].agg(['mean', 'median']).reset_index()
prob_stats['income_bracket'] = pd.Categorical(prob_stats['income_bracket'], categories=['High', 'Middle', 'Low'], ordered=True)
print(prob_stats.sort_values('income_bracket').to_string(index=False))

print("\n--- CHECK 3: EQUAL OPPORTUNITY TPR (BLIND MODEL) ---")
tpr_results = defaulters.groupby('income_bracket').agg(
    total_actual_defaults=('is_default', 'count'),
    correctly_predicted_defaults=('cv_pred_class', 'sum')
).reset_index()

tpr_results['recall_tpr'] = (tpr_results['correctly_predicted_defaults'] / tpr_results['total_actual_defaults']) * 100
tpr_results['income_bracket'] = pd.Categorical(tpr_results['income_bracket'], categories=['High', 'Middle', 'Low'], ordered=True)
print(tpr_results.sort_values('income_bracket').to_string(index=False, float_format="%.2f%%"))

print("\n--- CHECK 2: TPR CONTROLLED BY FICO DECILE ---")
# 0 = Lowest FICO tier, 9 = Highest FICO tier
defaulters['fico_decile'] = pd.qcut(defaulters['fico_range_low'], 10, labels=False, duplicates='drop')
fico_tpr = defaulters.groupby(['fico_decile', 'income_bracket']).agg(
    total=('is_default', 'count'),
    caught=('cv_pred_class', 'sum')
).reset_index()

fico_tpr['tpr'] = fico_tpr['caught'] / fico_tpr['total']
pivot_tpr = fico_tpr.pivot(index='fico_decile', columns='income_bracket', values='tpr')
pivot_tpr = pivot_tpr[['High', 'Middle', 'Low']]

print("TPR by FICO Decile within True Defaulters:")
# Fixed for modern pandas versions
print(pivot_tpr.map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "NaN"))
print("="*50 + "\n")