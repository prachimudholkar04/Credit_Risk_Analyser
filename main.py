# credit_risk_project.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Preprocessing Function
def preprocess(df, is_train=True, label_encoder_dict=None):
    df = df.copy()
    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

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

    if is_train:
        le_target = LabelEncoder()
        df['Loan_Status'] = le_target.fit_transform(df['Loan_Status'])
    else:
        le_target = None

    df = pd.get_dummies(df, columns=['Dependents', 'Property_Area'], drop_first=True)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['DebtIncomeRatio'] = df['LoanAmount'] / (df['TotalIncome'] + 1)

    return df, label_encoder_dict, le_target

# Load data
train_df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
test_df = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")
train_df, le_dict, le_target = preprocess(train_df, is_train=True)
test_df, _, _ = preprocess(test_df, is_train=False, label_encoder_dict=le_dict)

X = train_df.drop("Loan_Status", axis=1)
y = train_df["Loan_Status"]
X_test_final = test_df.reindex(columns=X.columns, fill_value=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss')
}

X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

results = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [], "ROC AUC": []}
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

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison_2.png")

results_df = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
print(results_df)

# Select best model based on ROC AUC
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_model.fit(X_scaled, y)
print(f"✅ Best Model selected: {best_model_name}")

# SHAP Explainability
explainer = shap.Explainer(best_model, X_scaled)
shap_values = explainer(X_scaled[:100])
shap.summary_plot(shap_values, X[:100], plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png")

# Hyperparameter Tuning Example
param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 10]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='roc_auc')
grid.fit(X_scaled, y)
print("Best Hyperparameters:", grid.best_params_)

# Cross-Validation
cv_scores = cross_val_score(RandomForestClassifier(), X_scaled, y, cv=5, scoring='roc_auc')
print("Cross-Validated ROC AUC:", cv_scores.mean())

# Save artifacts
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# Predict on external test set
y_pred_test = best_model.predict(X_test_scaled)
pd.DataFrame({"Prediction": y_pred_test}).to_csv("test_predictions.csv", index=False)

# Ensembling
ensemble = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier()),
    ('xgb', xgb.XGBClassifier(eval_metric='logloss')),
    ('lr', LogisticRegression())
], voting='soft')
ensemble.fit(X_scaled, y)
print("Ensemble Accuracy:", accuracy_score(y, ensemble.predict(X_scaled)))
