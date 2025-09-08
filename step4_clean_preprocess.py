# STEP 4: Data Cleaning and Preprocessing

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("/storage/emulated/0/PythonProjects/LoanRiskAI/train.csv")

# A. Check missing values
print("Missing values before cleaning:\n", df.isnull().sum())

# B. Fill missing values
# Fill categorical columns with mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Fill numerical columns with median
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)

# C. Drop unneeded columns
df.drop('Loan_ID', axis=1, inplace=True)

# D. Encode categorical variables
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# E. Final check
print("\nSample cleaned data:\n", df.head())
print("\nMissing values after cleaning:\n", df.isnull().sum())