import pandas as pd

# Load dataset
df = pd.read_csv("C:/School Files/Depression Predictor Model/Datasets/outlier_treatment.csv")

# Drop the 'Outlier_IQR' column if it exists
cols_to_drop = ['Outlier_IQR']
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]

if existing_cols_to_drop:
    df.drop(columns=existing_cols_to_drop, inplace=True)

# Round 'Study Satisfaction' if column exists
if 'Study Satisfaction' in df.columns:
    df['Study Satisfaction'] = df['Study Satisfaction'].round().astype(int)

# Display updated dataframe info
print(df.info())
print(df.head())

# Save updated dataframe
df.to_csv('C:/School Files/Depression Predictor Model/Datasets/dataset_standardization.csv', index=False)
