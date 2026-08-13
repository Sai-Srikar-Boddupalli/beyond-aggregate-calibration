import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb

print("Loading expanded dataset...")
cols = [
    'loan_amnt', 'annual_inc', 'int_rate', 'dti', 
    'fico_range_low', 'revol_util', 'pub_rec_bankruptcies', 'emp_length',
    'purpose', 'home_ownership', 'term', 'loan_status'
]

df = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols=cols, low_memory=False)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['is_default'] = (df['loan_status'] == 'Charged Off').astype(int)

# --- Feature Engineering: Missingness Flags ---
print("Applying row-wise missingness flags...")
missing_flag_cols = ["pub_rec_bankruptcies", "revol_util"]
for col in missing_flag_cols:
    df[f"{col}_missing"] = df[col].isna().astype(int)

# Handle emp_length specifically due to "n/a" text
df['emp_length'] = df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float)
df['emp_length_missing'] = df['emp_length'].isna().astype(int)

# Create Income Brackets for Stratification
conditions = [
    (df['annual_inc'] < 50000),
    (df['annual_inc'] >= 50000) & (df['annual_inc'] < 100000),
    (df['annual_inc'] >= 100000)
]
df['income_bracket'] = np.select(conditions, ['Low', 'Middle', 'High'], default='Unknown')

initial_len = len(df)
df = df[df['income_bracket'] != 'Unknown'].copy()
print(f"Dropped {initial_len - len(df)} rows with missing income.")

df['stratify_key'] = df['is_default'].astype(str) + "_" + df['income_bracket']

# --- Class Imbalance ---
pos_ratio = (len(df) - df['is_default'].sum()) / df['is_default'].sum()
print(f"Calculated scale_pos_weight (Negative/Positive ratio): {pos_ratio:.2f}")

# --- Pipeline Architecture ---
print("Building the strict preprocessor pipeline...")
categorical_cols = ["purpose", "home_ownership", "term"]
numeric_zeros = ["loan_amnt", "annual_inc", "int_rate", "dti", "fico_range_low", "revol_util", "pub_rec_bankruptcies"] + [f"{c}_missing" for c in missing_flag_cols]
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

# --- Stratified K-Fold Execution ---
print("\nExecuting Stratified 5-Fold Cross-Validation... (This will take time)")
X = df.drop(columns=['is_default', 'loan_status', 'stratify_key', 'income_bracket'])
y = df['is_default']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['cv_pred_prob'] = cross_val_predict(model_pipeline, X, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]

# --- Post-Run Audits ---
print("\n--- CALIBRATION CHECK ---")
plt.figure(figsize=(10, 6))
for bracket in ['Low', 'Middle', 'High']:
    mask = df['income_bracket'] == bracket
    brier = brier_score_loss(df[mask]['is_default'], df[mask]['cv_pred_prob'])
    print(f"{bracket} Income Brier Score (Lower is better): {brier:.4f}")
    
    prob_true, prob_pred = calibration_curve(df[mask]['is_default'], df[mask]['cv_pred_prob'], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'{bracket} Income')

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')
plt.title('Calibration Curve by Income Bracket')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.savefig('calibration_curve.png')
print("Saved calibration curve plot to 'calibration_curve.png'")

print("\n--- VISUALIZING THE MECHANISM ---")
print("Plotting probability distributions for True Defaulters...")
defaulters = df[df['is_default'] == 1].copy()

plt.figure(figsize=(12, 7))
sns.kdeplot(
    data=defaulters, 
    x='cv_pred_prob', 
    hue='income_bracket', 
    common_norm=False, 
    fill=True, 
    alpha=0.3, 
    linewidth=2,
    hue_order=['High', 'Middle', 'Low'],
    palette={'High': 'red', 'Middle': 'green', 'Low': 'blue'}
)
plt.axvline(x=0.15, color='black', linestyle='--', label='0.15 Discard Threshold')
plt.title('Predicted Probability Distribution for ACTUAL Defaulters', fontsize=14)
plt.xlabel('Model Predicted Probability of Default', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0, 1)
plt.legend(title='Income Bracket')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('defaulter_probability_histogram.png', dpi=300)
print("Saved visualization to 'defaulter_probability_histogram.png'")

# --- FIXED THRESHOLD SENSITIVITY AUDIT (TRUE DEFAULTERS ONLY) ---
print("\n" + "="*50)
print("--- FIXED THRESHOLD SENSITIVITY AUDIT (TRUE DEFAULTERS ONLY) ---")
print(f"Total True Defaulters (N): {len(defaulters)}")

for threshold in [0.10, 0.15, 0.20]:
    print(f"\nThreshold: {threshold}")
    defaulters['is_discarded'] = (defaulters['cv_pred_prob'] < threshold).astype(int)
    
    audit_results = defaulters.groupby('income_bracket').agg(
        total_defaults=('is_default', 'count'),
        discarded_records=('is_discarded', 'sum')
    ).reset_index()
    
    audit_results['percent_discarded'] = (audit_results['discarded_records'] / audit_results['total_defaults']) * 100
    print(audit_results[['income_bracket', 'total_defaults', 'discarded_records', 'percent_discarded']].sort_values('percent_discarded', ascending=False).to_string(index=False))

# --- MULTICOLLINEARITY CHECK ---
print("\n--- CORRELATION MATRIX (DEFAULTERS ONLY) ---")
corr_cols = ['annual_inc', 'fico_range_low', 'dti']
print(defaulters[corr_cols].corr().to_string())

print("\n" + "="*50)
print("--- FAIRNESS AUDIT: EQUAL OPPORTUNITY (HARDT ET AL. 2016) ---")
# Focus strictly on true defaulters to measure Recall (True Positive Rate)
defaulters = df[df['is_default'] == 1].copy()

# The missing value check for bookkeeping transparency
print("--- MISSING VALUE CHECK (Raw Data before Imputation) ---")
print(defaulters[['loan_amnt', 'dti', 'fico_range_low', 'emp_length', 'home_ownership', 'purpose', 'term']].isna().sum())
print("-" * 50)

# Calculate the predicted class using a standard 0.5 threshold
defaulters['cv_pred_class'] = (defaulters['cv_pred_prob'] > 0.5).astype(int)

# Group by income to calculate True Positive Rate (Recall) and False Negative Rate
tpr_results = defaulters.groupby('income_bracket').agg(
    total_actual_defaults=('is_default', 'count'),
    correctly_predicted_defaults=('cv_pred_class', 'sum')
).reset_index()

tpr_results['recall_tpr'] = (tpr_results['correctly_predicted_defaults'] / tpr_results['total_actual_defaults'])
tpr_results['miss_rate_fnr'] = 1 - tpr_results['recall_tpr']

# Format for clean output
tpr_results['recall_tpr'] = (tpr_results['recall_tpr'] * 100).round(2).astype(str) + '%'
tpr_results['miss_rate_fnr'] = (tpr_results['miss_rate_fnr'] * 100).round(2).astype(str) + '%'

# Ensure logical ordering
tpr_results['income_bracket'] = pd.Categorical(tpr_results['income_bracket'], categories=['High', 'Middle', 'Low'], ordered=True)
tpr_results = tpr_results.sort_values('income_bracket')

print("\nRecall (TPR) evaluates what percentage of actual defaults the model successfully caught.")
print(tpr_results.to_string(index=False))
print("="*50 + "\n")