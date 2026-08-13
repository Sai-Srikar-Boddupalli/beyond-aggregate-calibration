import pandas as pd
import statsmodels.formula.api as smf

print("Loading saved cross-validation results...")
try:
    df = pd.read_csv("cv_results_with_probs.csv")
    print("Successfully loaded cv_results_with_probs.csv")
except FileNotFoundError:
    print("ERROR: Could not find 'cv_results_with_probs.csv'. Make sure you are in the correct directory.")
    exit()

defaulters = df[df['is_default'] == 1].copy()
defaulters['income_bracket'] = pd.Categorical(
    defaulters['income_bracket'], 
    categories=['Low', 'Middle', 'High'], 
    ordered=True
)

print("\nFitting OLS Regression on TRUE DEFAULTERS...")
ols_model = smf.ols(formula="cv_pred_prob ~ C(income_bracket) + loan_amnt + dti + fico_range_low + emp_length", data=defaulters)
results = ols_model.fit()

print("\n" + "="*50)
print("--- REGRESSION RESULTS (TRUE DEFAULTERS ONLY) ---")
print(results.summary().tables[1])
print("="*50 + "\n")