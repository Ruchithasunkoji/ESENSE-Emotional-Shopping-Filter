# Step 1: Dataset Exploration

import pandas as pd

# Load dataset
df = pd.read_csv('manipulative_nonmanipulative_allproducts_1000_unique.csv')

# Display basic info
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample Data:")
print(df.head())

# Check for nulls
print("\nNull Values:")
print(df.isnull().sum())

# Check label distribution if labels are present
if 'emotion' in df.columns:
    print("\nEmotion Label Distribution:")
    print(df['emotion'].value_counts())

if 'manipulation_tactic' in df.columns:
    print("\nManipulation Tactic Distribution:")
    print(df['manipulation_tactic'].value_counts())
