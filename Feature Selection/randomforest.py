import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("C:/School Files/Depression Predictor Model/Datasets/dataset_standardization.csv")

# Drop ID column if exists
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Split into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode target if categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y_encoded)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for easy viewing
feat_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print feature importances
print("Feature Importances:")
print(feat_importances)

# Plot feature importances
plt.figure(figsize=(10,6))
plt.barh(feat_importances['Feature'], feat_importances['Importance'])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()

# Select features above a threshold (e.g., importance > 0.05)
threshold = 0.05
selected_features = feat_importances[feat_importances['Importance'] > threshold]['Feature'].tolist()

print("\n✅ Selected Features (Importance > 0.05):")
print(selected_features)

# Create a DataFrame with only selected features and the target
X_selected = X[selected_features]
y_selected = y_encoded

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y_selected)

# Reconstruct final balanced DataFrame
balanced_data = pd.DataFrame(X_balanced, columns=selected_features)
balanced_data["Depression"] = label_encoder.inverse_transform(y_balanced)

# Save balanced dataset
balanced_data.to_csv("C:/School Files/Depression Predictor Model/Model Testing/rf_smote_balanced.csv", index=False)

print("📁 Balanced dataset saved with shape:", balanced_data.shape)
