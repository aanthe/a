import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Age": [20, 22, 25, 28, 30, 32, 35, 40],
    "Score": [55, 60, 65, 70, 72, 78, 85, 90],
    "Salary": [25, 28, 32, 35, 40, 45, 50, 60]
}

df = pd.DataFrame(data)

plt.hist(df["Age"], bins=5)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.show()

plt.scatter(df["Age"], df["Salary"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
