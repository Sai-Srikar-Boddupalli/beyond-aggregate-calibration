import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import shap

print("Loading data for SHAP analysis...")
cols = [
    'loan_amnt', 'annual_inc', 'int_rate', 'dti', 
    'fico_range_low', 'revol_util', 'pub_rec_bankruptcies', 'emp_length',
    'purpose', 'home_ownership', 'term', 'loan_status'
]

df = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols=cols, low_memory=False)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['is_default'] = (df['loan_status'] == 'Charged Off').astype(int)

for col in ["pub_rec_bankruptcies", "revol_util"]:
    df[f"{col}_missing"] = df[col].isna().astype(int)
df['emp_length'] = df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float)
df['emp_length_missing'] = df['emp_length'].isna().astype(int)

# Post-Hoc Grouping
conditions = [
    (df['annual_inc'] < 50000),
    (df['annual_inc'] >= 50000) & (df['annual_inc'] < 100000),
    (df['annual_inc'] >= 100000)
]
df['income_bracket'] = np.select(conditions, ['Low', 'Middle', 'High'], default='Unknown')
df = df[df['income_bracket'] != 'Unknown'].copy()

pos_ratio = (len(df) - df['is_default'].sum()) / df['is_default'].sum()

print("Fitting the BLIND model on the full dataset...")
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

# Fit preprocessor to get feature names and transform X
X_raw = df.drop(columns=['is_default', 'loan_status', 'income_bracket', 'annual_inc'])
y = df['is_default']

X_transformed = preprocessor.fit_transform(X_raw)
feature_names = preprocessor.get_feature_names_out()

# Fit XGBoost directly on transformed data
model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_ratio, n_jobs=-1)
model.fit(X_transformed, y)

print("Calculating SHAP values for True Defaulters... (This may take a minute)")
# Isolate True Defaulters and their matching income brackets
defaulter_idx = df.index[df['is_default'] == 1]
X_defaulters = X_transformed[df['is_default'].values == 1]
income_labels = df.loc[defaulter_idx, 'income_bracket'].values

# Explain the XGBoost model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_defaulters)

# Create a DataFrame of the SHAP values
shap_df = pd.DataFrame(shap_values, columns=feature_names)

print("\n" + "="*70)
print("--- SHAP PROXY MECHANISM (Mean Absolute Impact on Log-Odds) ---")

# Take absolute values first, then group by the external labels array
abs_shap_df = shap_df.abs()
mean_shap = abs_shap_df.groupby(income_labels).mean().T

# Compare the extremes
mean_shap = mean_shap[['High', 'Low']] 

# Calculate the difference in how much the model relies on these features between groups
mean_shap['Absolute_Difference'] = (mean_shap['High'] - mean_shap['Low']).abs()
mean_shap = mean_shap.sort_values(by='Absolute_Difference', ascending=False).head(10)

print(mean_shap.round(4).to_string())
print("="*70 + "\n")