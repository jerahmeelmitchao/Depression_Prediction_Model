import pandas as pd

# === Step 1: Load Data ===
dataset = pd.read_csv('C:/School Files/Depression Predictor Model/Datasets/merge_depression_dataset.csv')

# Count duplicate rows
dup_count = dataset.duplicated().sum()
print("Duplicate Count in dataset: ", dup_count)

# Display duplicate rows (if needed)
dup_rows = dataset[dataset.duplicated()]

# Remove duplicates
without_dups = dataset.drop_duplicates()

# Double-check: count duplicates again in the cleaned dataset
cleaned_dup_count = without_dups.duplicated().sum()
print("Duplicate Count after cleaning: ", cleaned_dup_count)

# Save cleaned dataset
without_dups.to_csv('C:/School Files/Depression Predictor Model/Datasets/dataset_without_dups.csv', index=False)
