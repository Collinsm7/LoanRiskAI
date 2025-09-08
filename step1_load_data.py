import pandas as pd

#  path
df = pd.read_csv('/storage/emulated/0/PythonProjects/LoanRiskAI/data/train.csv')

# Preview data
print(df.head())
print(df.info())