# Step 5: Exploratory Data Analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv('/storage/emulated/0/PythonProjects/LoanRiskAI/train.csv')

# Set Seaborn theme
sns.set(style='whitegrid', palette='muted')

# 1. Loan Status Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Loan_Status')
plt.title('Loan Approval Status Count')
plt.savefig('/storage/emulated/0/PythonProjects/LoanRiskAI/eda1_loan_status_count.png')
plt.close()

# 2. Loan Status by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Gender', hue='Loan_Status')
plt.title('Loan Status by Gender')
plt.savefig('/storage/emulated/0/PythonProjects/LoanRiskAI/eda2_loan_status_by_gender.png')
plt.close()

# 3. Loan Status by Credit History
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Credit_History', hue='Loan_Status')
plt.title('Loan Status by Credit History')
plt.savefig('/storage/emulated/0/PythonProjects/LoanRiskAI/eda3_loan_status_by_credit_history.png')
plt.close()

# 4. Distribution of Applicant Income
plt.figure(figsize=(6, 4))
sns.histplot(df['ApplicantIncome'], kde=True, bins=30)
plt.title('Distribution of Applicant Income')
plt.savefig('/storage/emulated/0/PythonProjects/LoanRiskAI/eda4_applicant_income_distribution.png')
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('/storage/emulated/0/PythonProjects/LoanRiskAI/eda5_correlation_heatmap.png')
plt.close()

print("✅ Step 5 complete! All visualizations saved in /PythonProjects/LoanRiskAI/")