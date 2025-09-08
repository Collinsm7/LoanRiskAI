import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/storage/emulated/0/PythonProjects/LoanRiskAI/train.csv')

# Set the visual style
sns.set(style="whitegrid")

# 1. Countplot for Loan Status
sns.countplot(data=df, x='Loan_Status')
plt.title('Loan Approval Status Count')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

# 2. Countplot for Gender
sns.countplot(data=df, x='Gender')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 3. Countplot for Education
sns.countplot(data=df, x='Education')
plt.title('Education Level Distribution')
plt.xlabel('Education')
plt.ylabel('Count')
plt.show()

# 4. Countplot for Property Area
sns.countplot(data=df, x='Property_Area')
plt.title('Property Area Distribution')
plt.xlabel('Property Area')
plt.ylabel('Count')
plt.show()

# 5. Countplot for Credit History
sns.countplot(data=df, x='Credit_History')
plt.title('Credit History Distribution')
plt.xlabel('Credit History')
plt.ylabel('Count')
plt.show()

# 6. Boxplot: ApplicantIncome by Loan Status
sns.boxplot(data=df, x='Loan_Status', y='ApplicantIncome')
plt.title('Applicant Income by Loan Status')
plt.show()

# 7. Boxplot: LoanAmount by Loan Status
sns.boxplot(data=df, x='Loan_Status', y='LoanAmount')
plt.title('Loan Amount by Loan Status')
plt.show()

# 8. Boxplot: Total Income by Loan Status (Applicant + Coapplicant)
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
sns.boxplot(data=df, x='Loan_Status', y='Total_Income')
plt.title('Total Income by Loan Status')
plt.show()