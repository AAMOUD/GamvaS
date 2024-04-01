import pandas as pd

# Load the data
data = pd.read_csv('/data.csv')  # Update this to the path of your CSV file

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Drop rows with missing values (optional)
data = data.dropna()

# Assume your CSV has 'income', 'expenses', and 'savings_goal' columns
features = data[['income', 'expenses']]
labels = data['savings_goal']

print("Features:\n", features.head())
print("Labels:\n", labels.head())
