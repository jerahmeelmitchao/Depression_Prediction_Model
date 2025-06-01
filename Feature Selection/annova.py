import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("C:/School Files/Depression Predictor Model/Datasets/dataset_standardization.csv")

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

feature_names = X.columns

f_values = []
p_values = []
significance = []
mean_values_str = []

# For each feature, run ANOVA and calculate mean values per class
for feature in feature_names:
    groups = [X[feature][y_encoded == label] for label in np.unique(y_encoded)]
    f_stat, p_value = f_oneway(*groups)

    f_values.append(f_stat)
    p_values.append(p_value)
    significance.append("Significant" if p_value < 0.05 else "Not Significant")
    means = [f"{label_encoder.inverse_transform([label])[0]}: {group.mean():.3f}" for label, group in zip(np.unique(y_encoded), groups)]
    mean_values_str.append(", ".join(means))

# Create result DataFrame
result = pd.DataFrame({
    "Feature": feature_names,
    "Mean Values per Class": mean_values_str,
    "Significance": significance,
    "p_value": p_values
})

# Filter only significant features (p < 0.05)
significant_features = result[result['p_value'] < 0.05]['Feature'].tolist()

# Subset data to significant features + target
data_significant = data[significant_features + [data.columns[-1]]]

# Separate features and target
X_sig = data_significant[significant_features]
y_sig = label_encoder.fit_transform(data_significant[data.columns[-1]])

# Apply SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_sig, y_sig)

# Combine into final DataFrame
balanced_data = pd.DataFrame(X_balanced, columns=significant_features)
balanced_data["Depression"] = label_encoder.inverse_transform(y_balanced)

# Save the balanced dataset
balanced_data.to_csv("C:/School Files/Depression Predictor Model/Model Testing/annova_smote_balanced.csv", index=False)

print("Balanced dataset saved with shape:", balanced_data.shape)
