import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

print("Loading the clean data... (Grabbing just 4 basic columns)")

# 1. We only load the columns we need right now so it runs fast
columns_to_use = ['loan_amnt', 'annual_inc', 'int_rate', 'dti', 'is_default']
df = pd.read_csv("baseline_data.csv", usecols=columns_to_use)

# 2. Convert text back to numbers (since we saved everything as text in step 1 to avoid crashes)
for col in columns_to_use:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows that are missing these 4 basic pieces of info
df = df.dropna()

# 3. Separate our data into "The Questions" (X) and "The Answer" (y)
X = df[['loan_amnt', 'annual_inc', 'int_rate', 'dti']]
y = df['is_default']

# 4. Split the data: 80% to study and learn, 20% to take a test at the end
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the baseline XGBoost model... (This will take a few seconds)")
# 5. Build and train the model
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

print("Grading the model's test...")
# 6. Make predictions on the 20% test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# 7. Score the results
accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print("\n--- BASELINE RESULTS ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc:.4f}")