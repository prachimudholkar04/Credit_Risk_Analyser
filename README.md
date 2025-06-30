# 🏦 Credit Risk Analysis with Machine Learning

## 📌 Project Overview

Credit Risk Analysis is the process of evaluating the likelihood that a borrower may default on their loan. This project applies various machine learning algorithms to predict loan default risk based on financial and demographic data.

## ❓ Why

Financial institutions need robust tools to:
- Minimize loan defaults
- Improve decision-making
- Automate risk profiling for loan applicants

Machine learning enables data-driven predictions by analyzing historical patterns in applicant data.

## 💡 What

This project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing & feature engineering
- Training & evaluation of multiple ML models
- Model comparison based on Accuracy, Precision, Recall, F1-Score, and ROC AUC
- Visualizations of metrics and ROC curves
- SHAP (SHapley Additive Explanations) for interpretability

## ⚙️ How

### 📁 Project Structure
credit-risk-analysis/
├── data/ # Dataset files
├── models/ # Trained models (e.g., .pkl)
├── outputs/ # Evaluation plots and results
├── app/ # Streamlit or Flask app (optional)
├── notebooks/ # EDA and experimentation
├── scripts/ # Modular Python scripts
│ ├── preprocessing.py
│ ├── model_training.py
│ └── evaluate_models.py
├── README.md
└── requirements.txt
### 🧪 Machine Learning Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- XGBoost

### 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### 📈 Visualizations
- ROC Curve for all models
- Bar plots for metric comparison
- Correlation heatmap
- Feature importance (via Random Forest & SHAP)

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
## 📈 Results

All models were evaluated using Accuracy, Precision, Recall, F1 Score, and ROC AUC. Among them:

### 🏆 Best Model: **Random Forest Classifier**
- **Accuracy**: 82.1%
- **F1 Score**: 86.5%
- **ROC AUC**: 88.6%

This model demonstrated the best balance between false positives and false negatives, with strong generalization on unseen test data.

Other models like **XGBoost** and **Logistic Regression** also performed well and can be considered if further tuning or ensemble strategies are pursued.

## 🧠 Insights
- **Credit History** and **Debt-to-Income Ratio** were among the most influential features.
- Tree-based models benefit from the non-linearity and feature interactions present in financial datasets.
- SHAP analysis confirmed that features like `Credit_History`, `TotalIncome`, and `LoanAmount` contribute most significantly to model decisions.

