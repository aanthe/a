"""
Purpose:
This file shows how to inspect simple data with graphs.
We want to understand the shape of the data, compare variables, and see relationships visually.

What we are trying to achieve:
Turn raw numbers into pictures so patterns become easier to notice.

Overall flow:
1. Create a small example table of data.
2. Draw a histogram to see how one column is distributed.
3. Draw a scatter plot to compare two columns.
4. Draw a correlation heatmap to see which columns move together.
"""

Theory:
Graphs help us convert numbers into patterns the eye can understand quickly.
Histogram shows distribution, scatter plot shows relationship, and heatmap shows correlation strength.

Viva:
Q: Why use a histogram?
A: To see how values are spread across ranges.
Q: Why use a scatter plot?
A: To check whether two variables move together.
Q: What does a correlation heatmap show?
A: How strongly columns are related.

Output:
1. A histogram of Age.
2. A scatter plot of Age vs Salary.
3. A heatmap showing correlations between the columns.

import pandas as pd  # pandas helps us store tabular data in rows and columns, like a spreadsheet.
import matplotlib.pyplot as plt  # matplotlib draws graphs from our data.
import seaborn as sns  # seaborn builds prettier statistical plots on top of matplotlib.

# Term: dataset means a collection of related data values we want to study together.
# Create sample dataset.
# Why: before drawing plots, we need numbers to visualize.
data = {
    "Age": [20, 22, 25, 28, 30, 32, 35, 40],
    "Score": [55, 60, 65, 70, 72, 78, 85, 90],
    "Salary": [25, 28, 32, 35, 40, 45, 50, 60]
}

# Term: DataFrame means a table with rows and columns, like an Excel sheet in code.
df = pd.DataFrame(data)  # Convert the dictionary into a table so each column can be analyzed easily.

# Term: histogram means a bar-style chart that groups numbers into ranges and counts how many values fall in each range.
# Histogram.
# Why: a histogram groups values into ranges so we can see how the data is spread.
plt.hist(df["Age"], bins=5)  # Split ages into 5 buckets.
plt.xlabel("Age")  # Label the horizontal axis so we know what the bars represent.
plt.ylabel("Frequency")  # Frequency means how many values fall into each bucket.
plt.title("Age Distribution")  # Title tells the viewer the purpose of this plot.
plt.show()  # Display the plot window.

# Term: scatter plot means a chart where each dot is one pair of related values.
# Scatter plot.
# Why: a scatter plot helps us check whether two variables move together.
plt.scatter(df["Age"], df["Salary"])  # Each dot is one person/data row.
plt.xlabel("Age")  # Age is the input variable on the x-axis.
plt.ylabel("Salary")  # Salary is the output variable on the y-axis.
plt.title("Age vs Salary")  # We name the relationship we are checking.
plt.show()  # Show the scatter plot.

# Term: correlation means how strongly two variables move together.
# Term: heatmap means a color-based table where color intensity shows a value.
# Correlation heatmap.
# Why: correlation shows whether two columns tend to increase/decrease together.
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")  # annot=True prints numbers; cmap colors strong/weak relationships.
plt.title("Correlation Heatmap")  # Title explains that the colors encode correlation values.
plt.show()  # Display the heatmap.
