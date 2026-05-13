import pandas as pd

# Load data
df = pd.read_csv(r'D:\creditcard.csv')

# Check class distribution
print("Class Distribution:")
print(df['Class'].value_counts())

# Percentage calculation
print("\nPercentage:")
print(df['Class'].value_counts(normalize=True) * 100)

# Basic statistics for Amount
print("\nAmount Statistics:")
print(df['Amount'].describe())
