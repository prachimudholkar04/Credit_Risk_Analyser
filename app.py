# app.py

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load model, scaler, and feature columns
model = joblib.load("best_model.pkl")

if os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")
else:
    st.error("❌ Scaler file not found. Make sure 'scaler.pkl' exists in the project folder.")
    st.stop()

if os.path.exists("feature_columns.pkl"):
    feature_columns = joblib.load("feature_columns.pkl")
else:
    st.error("❌ Feature column list not found. Make sure 'feature_columns.pkl' exists in the project folder.")
    st.stop()

st.title("📊 Credit Risk Prediction App")
st.write("Enter applicant details to predict loan approval.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
app_income = st.number_input("Applicant Income", min_value=0)
coapp_income = st.number_input("Coapplicant Income", min_value=0)
loan_amt = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=1)
credit_history = st.selectbox("Credit History", [1, 0])
dependents_1 = st.checkbox("1 Dependent")
dependents_2 = st.checkbox("2 Dependents")
dependents_3p = st.checkbox("3+ Dependents")
prop_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Prepare input row
input_data = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": app_income,
    "CoapplicantIncome": coapp_income,
    "LoanAmount": loan_amt,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Dependents_1": int(dependents_1),
    "Dependents_2": int(dependents_2),
    "Dependents_3+": int(dependents_3p),
    "Property_Area_Rural": int(prop_area == "Rural"),
    "Property_Area_Semiurban": int(prop_area == "Semiurban"),
    "Property_Area_Urban": int(prop_area == "Urban"),
    "TotalIncome": app_income + coapp_income,
    "DebtIncomeRatio": loan_amt / (app_income + coapp_income + 1)
}

input_df = pd.DataFrame([input_data])

# Align columns to match training
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Scaler error: {e}")
    st.stop()

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    st.write("Raw prediction value:", pred)  # Debug info
    if pred == 1:
        st.success("🚀 Approved")
    else:
        st.error("❌ Not Approved")
        st.warning("🔍 Reason: Application appears to have high risk based on income, credit history, or loan details.")
