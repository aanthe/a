import pandas as pd
import numpy as np

data = {
    "Age": [20, 25, np.nan, 30, 22],
    "Salary": [30000, 40000, 35000, np.nan, 32000],
    "Category": ["A", "B", "A", "C", "B"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

df["Age"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())
df["Salary"] = (df["Salary"] - df["Salary"].min()) / (df["Salary"].max() - df["Salary"].min())

df = pd.get_dummies(df, columns=["Category"], dtype=int)

print("\nPreprocessed Dataset:")
print(df)
