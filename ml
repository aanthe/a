# =====================================================
# STUDENT ADMISSION PREDICTION
# LINEAR REGRESSION VS RANDOM FOREST
# =====================================================

# 1. IMPORT LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import builtins

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# =====================================================
# 2. CREATE DATASET
# =====================================================

np.random.seed(42)

n = 100

data = {
    "Serial No.": range(1, n + 1),
    "GRE Score": np.random.randint(260, 341, n),
    "TOEFL Score": np.random.randint(80, 121, n),
    "University Rating": np.random.randint(1, 6, n),
    "SOP": np.round(np.random.uniform(1, 5, n), 1),
    "LOR": np.round(np.random.uniform(1, 5, n), 1),
    "CGPA": np.round(np.random.uniform(6, 10, n), 2),
    "Research": np.random.randint(0, 2, n)
}

df = pd.DataFrame(data)

# Generate admission chance
chance = (
    0.20
    + (df["GRE Score"] - 260) * 0.002
    + (df["TOEFL Score"] - 80) * 0.003
    + df["University Rating"] * 0.025
    + df["SOP"] * 0.015
    + df["LOR"] * 0.010
    + (df["CGPA"] - 6) * 0.07
    + df["Research"] * 0.04
    + np.random.normal(0, 0.04, n)
)

df["Chance of Admit"] = np.clip(chance, 0.20, 0.99)

# Save dataset
df.to_csv("Admission_Predict.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Dataset shape:", df.shape)


# =====================================================
# 3. LOAD AND CLEAN DATA
# =====================================================

df = pd.read_csv("Admission_Predict.csv")

df.columns = df.columns.str.strip()

df = df.drop("Serial No.", axis=1)

print("\nCleaned Dataset:")
print(df.head())


# =====================================================
# 4. VISUALIZE DATA
# =====================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlations")
plt.show()


# =====================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING
# =====================================================

X = df.drop("Chance of Admit", axis=1)

y = df["Chance of Admit"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =====================================================
# 6. MODEL A: LINEAR REGRESSION
# =====================================================

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)


# =====================================================
# 7. MODEL B: RANDOM FOREST REGRESSION
# =====================================================

rf = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)


# =====================================================
# 8. EVALUATE BOTH MODELS
# =====================================================

lr_score = r2_score(y_test, lr_pred)

rf_score = r2_score(y_test, rf_pred)

print("\nMODEL EVALUATION")

print(
    "Linear Regression R2 Score:",
    builtins.round(lr_score, 4)
)

print(
    "Random Forest R2 Score:",
    builtins.round(rf_score, 4)
)

print(
    "Linear Regression R2 (%):",
    builtins.round(lr_score * 100, 2), "%"
)

print(
    "Random Forest R2 (%):",
    builtins.round(rf_score * 100, 2), "%"
)


# =====================================================
# 9. PREDICT A NEW STUDENT'S ADMISSION CHANCE
# =====================================================

new_student = pd.DataFrame(
    [[320, 110, 4, 4.0, 4.0, 9.0, 1]],
    columns=X.columns
)

prediction = lr.predict(new_student)

print(
    "\nPredicted Admission Chance:",
    builtins.round(prediction[0] * 100, 2),
    "%"
)


# =====================================================
# 10. COMPARE MODEL PERFORMANCE
# =====================================================

models = ["Linear Regression", "Random Forest"]

scores = [lr_score, rf_score]

plt.figure(figsize=(7, 4))

plt.bar(models, scores)

plt.title("Model Performance Comparison")

plt.ylabel("R2 Score")

plt.show()


# =====================================================
# 11. DISPLAY FINAL RESULTS
# =====================================================

print("\nPROGRAM COMPLETED SUCCESSFULLY!")
