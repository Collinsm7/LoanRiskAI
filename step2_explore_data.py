import pandas as pd

# Load the data
df = pd.read_csv('/storage/emulated/0/PythonProjects/LoanRiskAI/train.csv')

# Basic shape
print("Shape of dataset:", df.shape)

# Column names
print("\nColumns:\n", df.columns.tolist())

# Quick summary
print("\nSummary stats:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values:\n")
print(df.isnull().sum())