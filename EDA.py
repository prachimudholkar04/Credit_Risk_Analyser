import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Re-load clean dataset to make sure it's fresh
eda_df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
# Create a list of EDA plots
eda_plots = []

# 1. Distribution of Applicant Income
plt.figure(figsize=(8, 5))
sns.histplot(eda_df['ApplicantIncome'], kde=True, bins=30)
plt.title("Distribution of Applicant Income")
plt.tight_layout()
plt.savefig("eda_applicant_income.png")
plt.close()
eda_plots.append(("Applicant Income Distribution", "eda_applicant_income.png"))

# 2. Boxplot: Loan Amount by Property Area
plt.figure(figsize=(8, 5))
sns.boxplot(x='Property_Area', y='LoanAmount', data=eda_df)
plt.title("Loan Amount vs Semiurban Property Area")
plt.tight_layout()
plt.savefig("eda_loanamount_property_area.png")
plt.close()
eda_plots.append(("Loan Amount by Property Area", "eda_loanamount_property_area.png"))

# 3. Countplot: Education
plt.figure(figsize=(6, 4))
sns.countplot(x='Education', data=eda_df)
plt.title("Education Distribution")
plt.tight_layout()
plt.savefig("eda_education_distribution.png")
plt.close()
eda_plots.append(("Education Distribution", "eda_education_distribution.png"))

# 4. Countplot: Gender vs Loan Status
plt.figure(figsize=(8, 5))
sns.boxplot(x='Property_Area', y='LoanAmount', data=eda_df)
plt.title("Loan Amount by Property Area")
plt.tight_layout()
plt.show()

# 5. Scatter Plot: Debt-Income Ratio vs LoanAmount
plt.figure(figsize=(7, 5))
sns.scatterplot(x='DebtIncomeRatio', y='LoanAmount', hue='Loan_Status', data=eda_df)
plt.title("Debt-Income Ratio vs Loan Amount")
plt.tight_layout()
plt.savefig("eda_debt_income_scatter.png")
plt.close()
eda_plots.append(("Debt-Income Ratio vs Loan Amount", "eda_debt_income_scatter.png"))

# Show summary
eda_df_summary = pd.DataFrame(eda_plots, columns=["EDA Plot", "Image File"])