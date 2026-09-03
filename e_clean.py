"""
Purpose:
This file shows basic data cleaning and preprocessing steps.
We want to take messy raw data and turn it into a form that is easier for analysis or machine learning.

What we are trying to achieve:
Fix missing values, scale numeric values, and convert text categories into numbers.

Overall flow:
1. Create a small example dataset with missing values.
2. Inspect the original data.
3. Replace missing numeric values with the mean.
4. Normalize numeric columns to a 0 to 1 range.
5. Encode categorical text into separate numeric columns.
6. Display the cleaned result.
"""

Theory:
Raw data often contains missing values and text categories.
Cleaning makes data usable by replacing blanks, scaling numbers, and converting labels into numeric form.

Viva:
Q: Why fill missing values?
A: Many algorithms cannot handle blanks directly.
Q: Why normalize data?
A: To bring different numeric ranges onto a common scale.
Q: Why use one-hot encoding?
A: To convert categories into numbers without inventing false order.

Output:
The original dataset is printed first, then the cleaned and encoded dataset is printed after preprocessing.

import pandas as pd  # pandas gives us DataFrames, which are like neat tables.
import numpy as np  # NumPy gives us NaN for missing numbers and numeric helpers.

# Term: missing value means data that is absent or unknown in a table.
# Term: NaN means "Not a Number", a special marker used to represent missing numeric data.
# Create sample dataset.
# Why: cleaning only makes sense when we start from imperfect raw data.
data = {
    "Age": [20, 25, np.nan, 30, 22],
    "Salary": [30000, 40000, 35000, np.nan, 32000],
    "Category": ["A", "B", "A", "C", "B"]
}

df = pd.DataFrame(data)  # Put the raw data into a table.

# Term: preprocessing means preparing raw data so it is easier for a model to use.
print("Original Dataset:")  # Show the data before fixing it so we can compare before/after.
print(df)  # Display the raw rows, including missing values.

# Term: mean means the average value, found by adding values and dividing by how many there are.
# Handle missing values.
# Why: most machine learning algorithms cannot work well with blank cells, so we replace them.
df["Age"] = df["Age"].fillna(df["Age"].mean())  # Replace missing age with the average age.
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())  # Replace missing salary with the average salary.

# Term: normalization means scaling numbers into a common range.
# Term: min-max scaling means mapping values between 0 and 1 using the smallest and largest values.
# Min-Max normalization.
# Why: putting features on a 0-to-1 scale helps compare them fairly and helps some models train better.
df["Age"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())  # Scale age between 0 and 1.
df["Salary"] = (df["Salary"] - df["Salary"].min()) / (df["Salary"].max() - df["Salary"].min())  # Scale salary between 0 and 1.

# Term: categorical variable means a value that belongs to a group or label, not a numeric scale.
# Term: one-hot encoding means turning each category into a separate 0/1 column.
# Encode categorical variable.
# Why: computers need numbers, so text labels like A/B/C are converted into separate yes/no columns.
df = pd.get_dummies(df, columns=["Category"], dtype=int)  # Create one-hot columns for each category.

print("\nPreprocessed Dataset:")  # Show the cleaned and transformed version.
print(df)  # Display the final table after preprocessing.
