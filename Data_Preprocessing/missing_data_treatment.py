import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# === Load Data ===
# Load Data (replace '?' with NaN)
df = pd.read_csv('C:/School Files/Depression Predictor Model/Datasets/dataset_without_dups.csv', na_values='?')

    
# === Fill Missing Values ===
df['City'].fillna(df['City'].mode().iloc[0], inplace=True)
df['Work Pressure'] = df['Work Pressure'].fillna(method='ffill')
df['Sleep Duration'].fillna(df['Sleep Duration'].mode().iloc[0], inplace=True)
df['Work/Study Hours'].fillna(df['Work/Study Hours'].mode().iloc[0], inplace=True)
df['Family History of Mental Illness'].fillna(df['Family History of Mental Illness'].mode().iloc[0], inplace=True)

# Check again for missing values 
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)

# === Manual Encoding: Define mappings ===
gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
city_map = {val: i for i, val in enumerate(df['City'].dropna().unique())}
profession_map = {val: i for i, val in enumerate(df['Profession'].dropna().unique())}
sleep_duration_map = {val: i for i, val in enumerate(df['Sleep Duration'].dropna().unique())}
dietary_habits_map = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
degree_map = {val: i for i, val in enumerate(df['Degree'].dropna().unique())}
suicidal_map = {'No': 0, 'Yes': 1}
family_history_map = {'No': 0, 'Yes': 1}

# === Apply Mappings ===
df['Gender'] = df['Gender'].map(gender_map)
df['City'] = df['City'].map(city_map)
df['Profession'] = df['Profession'].map(profession_map)
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_duration_map)
df['Dietary Habits'] = df['Dietary Habits'].map(dietary_habits_map)
df['Degree'] = df['Degree'].map(degree_map)
df['Suicidal thoughts'] = df['Suicidal thoughts'].map(suicidal_map)
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(family_history_map)

# Regression
features = ['Gender', 'Age', 'City', 'Profession', 
            'Academic Pressure', 'Work Pressure', 'CGPA', 'Job Satisfaction','Sleep Duration',
            'Dietary Habits','Degree','Suicidal thoughts','Work/Study Hours','Financial Stress',
            'Family History of Mental Illness','Depression']

target = df['Study Satisfaction']

# Separate data into known and missing target
df_known = df.dropna(subset=['Study Satisfaction']) 
df_missing = df[df['Study Satisfaction'].isnull()]

# Define X (features) and y (target)
X_train = df_known[features]
y_train = df_known['Study Satisfaction']
X_missing = df_missing[features]

# Fill missing values in features using median from training data
X_train = X_train.fillna(X_train.median())
X_missing = X_missing.fillna(X_train.median())

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing target values
predicted_values = model.predict(X_missing)

# Assign predicted values back to missing rows
df.loc[df['Study Satisfaction'].isnull(), 'Study Satisfaction'] = predicted_values

df['Dietary Habits'].fillna(df['Dietary Habits'].mode().iloc[0], inplace=True)
df['Financial Stress'].fillna(df['Financial Stress'].mode().iloc[0], inplace=True)
# Check again for missing values
missing_values_after = df.isnull().sum()
print("Missing Values per Column (After Filling):")
print(missing_values_after)


# === Save Cleaned File ===
df.to_csv('C:/School Files/Depression Predictor Model/Datasets/missing_data_treatment.csv', index=False)

# === Save Mapping for Web App ===
import json

mapping_dict = {
    'Gender': gender_map,
    'City': city_map,
    'Profession': profession_map,
    'Sleep Duration': sleep_duration_map,
    'Dietary Habits': dietary_habits_map,
    'Degree': degree_map,
    'Suicidal thoughts': suicidal_map,
    'Family History of Mental Illness': family_history_map
}

with open('C:/School Files/Depression Predictor Model/category_mappings.json', 'w') as f:
    json.dump(mapping_dict, f, indent=4)

print("Done. Cleaned dataset and mapping file saved.")
