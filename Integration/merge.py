import pandas as pd

# Load uncleaned data
depression_survey1 = pd.read_csv("C:/School Files/Depression Predictor Model/Datasets/depression_survey1_uncleaned.csv")
depression_survey2 = pd.read_csv("C:/School Files/Depression Predictor Model/Datasets/depression_survey2_uncleaned.csv")

# Clean `id` column in both datasets
depression_survey1['id'] = depression_survey1['id'].astype(str).str.replace('"', '', regex=False).str.strip()
depression_survey2['id'] = depression_survey2['id'].astype(str).str.replace('"', '', regex=False).str.strip()

# Merge based on cleaned `id`
final_data = pd.merge(depression_survey1, depression_survey2, on='id', how='inner')

# Save merged dataset
final_data.to_csv("C:/School Files/Depression Predictor Model/Datasets/merge_depression_dataset.csv", index=False)
