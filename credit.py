# Full reset of all steps: loading, preprocessing, modeling with separate train/test files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt

# Define a preprocessing function to use for both train and test datasets
def preprocess(df, is_train=True, label_encoder_dict=None):
    df = df.copy()
    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)

    # Fill missing values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode binary categorical variables
    binary_cols = ['Gender', 'Married', 'Education', 'Self_Employed']
    if is_train:
        label_encoder_dict = {}
        for col in binary_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoder_dict[col] = le
    else:
        for col in binary_cols:
            le = label_encoder_dict[col]
            df[col] = le.transform(df[col])

    # Encode target if training
    if is_train:
        le_target = LabelEncoder()
        df['Loan_Status'] = le_target.fit_transform(df['Loan_Status'])
    else:
        le_target = None

    # One-hot encode multiclass categorical variables
    df = pd.get_dummies(df, columns=['Dependents', 'Property_Area'], drop_first=True)

    # Feature engineering
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['DebtIncomeRatio'] = df['LoanAmount'] / (df['TotalIncome'] + 1)

    return df, label_encoder_dict, le_target

# Load training and test datasets
df_train = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
df_test = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

# Preprocess both datasets
df_train_clean, le_dict, le_target = preprocess(df_train, is_train=True)
df_test_clean, _, _ = preprocess(df_test, is_train=False, label_encoder_dict=le_dict)

# Split into features and target
X_train = df_train_clean.drop("Loan_Status", axis=1)
y_train = df_train_clean["Loan_Status"]

# Align columns of test set with train set (for one-hot encoding compatibility)
X_test = df_test_clean.reindex(columns=X_train.columns, fill_value=0)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss')
}

# Store evaluation results
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": [],
    "ROC AUC": []
}

# Assume no ground truth for test set; simulate test evaluation using train split
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train_sub, y_train_sub)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)

    results["Model"].append(name)
    results["Accuracy"].append(accuracy_score(y_val, y_pred))
    results["Precision"].append(precision_score(y_val, y_pred))
    results["Recall"].append(recall_score(y_val, y_pred))
    results["F1 Score"].append(f1_score(y_val, y_pred))
    results["ROC AUC"].append(roc_auc_score(y_val, y_proba))

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    plt.plot(fpr, tpr, label=name)

# Plot ROC Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Display results
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by="ROC AUC", ascending=False)
print(results_df)

#Identify the Best Model

# Choose your main metric for selection (ROC AUC is common for classification)
main_metric = "ROC AUC"

# Get the row with the highest value for the main metric
best_model_row = results_df.loc[results_df[main_metric].idxmax()]

# Print best model statement
print("\nBest Model Based on", main_metric, ":")
print(f"{best_model_row['Model']} with {main_metric} = {best_model_row[main_metric]:.4f}")

#User-Selected Best Model
# ==============================

available_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

print("\nAvailable metrics for model selection:")
for i, metric in enumerate(available_metrics, start=1):
    print(f"{i}. {metric}")

# Prompt user
while True:
    user_input = input("\nEnter the metric name to select the best model: ").strip()
    if user_input in available_metrics:
        break
    else:
        print("❌ Invalid choice. Please select from the listed metrics.")

# Select and print best model based on user-selected metric
best_model_row = results_df.loc[results_df[user_input].idxmax()]
print(f"\n✅ Best Model Based on '{user_input}':")
print(f"{best_model_row['Model']} with {user_input} = {best_model_row[user_input]:.4f}")


print(df_test.columns)
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("Predicted Loan_Status (0 = No, 1 = Yes):")
print(y_pred)

# Optional: Save results
pd.DataFrame({"Prediction": y_pred}).to_csv("test_predictions.csv", index=False)