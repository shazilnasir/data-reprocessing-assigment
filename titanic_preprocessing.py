import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("train.csv")

# ----- Data Cleaning -----
# Example: Fill missing 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill 'Embarked' with mode (most frequent)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to too many missing values
df.drop(columns=['Cabin'], inplace=True)

# ----- Noisy Data Handling -----
# Example: Bin 'Age' into categories (young, middle, old)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 50, 100], labels=['Child', 'Adult', 'Senior'])

# ----- Data Integration -----
# Combine data if you had multiple datasets, but here we'll demonstrate by extracting 'Title' from 'Name'
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Show cleaned data
print(df.head())

# Save preprocessed data
df.to_csv("processed_titanic.csv", index=False)
