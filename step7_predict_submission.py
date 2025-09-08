# 📦 Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 📂 Load Data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 🧼 Clean missing values in train set
train_df = train_df.fillna({
    'Gender': train_df['Gender'].mode()[0],
    'Married': train_df['Married'].mode()[0],
    'Dependents': train_df['Dependents'].mode()[0],
    'Self_Employed': train_df['Self_Employed'].mode()[0],
    'Credit_History': train_df['Credit_History'].mode()[0],
    'LoanAmount': train_df['LoanAmount'].median(),
    'Loan_Amount_Term': train_df['Loan_Amount_Term'].mode()[0]
})

# 🧼 Clean missing values in test set
test_df = test_df.fillna({
    'Gender': test_df['Gender'].mode()[0],
    'Married': test_df['Married'].mode()[0],
    'Dependents': test_df['Dependents'].mode()[0],
    'Self_Employed': test_df['Self_Employed'].mode()[0],
    'Credit_History': test_df['Credit_History'].mode()[0],
    'LoanAmount': test_df['LoanAmount'].median(),
    'Loan_Amount_Term': test_df['Loan_Amount_Term'].mode()[0]
})

# 🔤 Label Encode Categorical Columns
cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
le = LabelEncoder()
for col in cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# 🎯 Separate features and target
X_train = train_df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y_train = train_df['Loan_Status'].map({'Y': 1, 'N': 0})  # Encode target

X_test = test_df.drop(['Loan_ID'], axis=1)

# 🤖 Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 🔮 Predict
predictions = model.predict(X_test)
final_predictions = np.where(predictions == 1, 'Y', 'N')

# 📄 Create submission DataFrame
submission = pd.DataFrame({
    'Loan_ID': test_df['Loan_ID'],
    'Loan_Status': final_predictions
})

# 💾 Save to CSV
submission.to_csv("loan_predictions_submission.csv", index=False)
print("✅ Submission file saved as 'loan_predictions_submission.csv'")