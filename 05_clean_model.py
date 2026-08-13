import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

print("Loading data and splitting into Study (Train) and Test groups...")
df = pd.read_csv("baseline_data.csv", usecols=['loan_amnt', 'annual_inc', 'int_rate', 'dti', 'is_default']).dropna()

X = df[['loan_amnt', 'annual_inc', 'int_rate', 'dti']]
y = df['is_default']

# Split the data FIRST. We keep the Test data completely untouched.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Original Training rows: {len(X_train):,}")
print("Hunting for noise in the Training data...")

# Find the noise just in the training set
model = xgb.XGBClassifier(eval_metric='logloss')
train_probs = cross_val_predict(model, X_train, y_train, cv=3, method='predict_proba')[:, 1]

# Create a filter: Keep the row as long as it is NOT a suspicious default
# (Meaning, drop it if it actually defaulted (1) but the model thought it was < 15% likely)
mask = ~((y_train == 1) & (train_probs < 0.15))

X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

print(f"Clean Training rows: {len(X_train_clean):,}")
print(f"Removed {len(X_train) - len(X_train_clean):,} confusing records.")

print("\nTraining the new model on the CLEAN data...")
model.fit(X_train_clean, y_train_clean)

print("Grading the new model on the UNTOUCHED messy test data...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print("\n--- CLEAN MODEL RESULTS ---")
print(f"Clean Accuracy: {accuracy:.4f}")
print(f"Clean AUC Score: {auc:.4f}")