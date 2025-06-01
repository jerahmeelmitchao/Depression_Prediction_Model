import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

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

# Define model and apply RFE
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X, y_encoded)

# Rank and visualize features
ranking = rfe.ranking_
feature_names = X.columns
selected_features = feature_names[rfe.support_]

print("\n📊 RFE Feature Ranking (lower is better):\n")
for name, rank in sorted(zip(feature_names, ranking), key=lambda x: x[1]):
    bar = "█" * (15 - rank + 1) if rank <= 15 else ""
    tag = "✅ Selected" if rank == 1 else "   "
    print(f"{name:<30} Rank: {rank:<2} {bar} {tag}")

# Create new DataFrame with selected features and apply SMOTE
X_selected = X[selected_features]
y_selected = y_encoded

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y_selected)

# Reconstruct and save balanced DataFrame
balanced_data = pd.DataFrame(X_balanced, columns=selected_features)
balanced_data["Depression"] = label_encoder.inverse_transform(y_balanced)

balanced_data.to_csv("C:/School Files/Depression Predictor Model/Model Testing/rfe_smote_balanced.csv", index=False)

print("\n📁 File saved: rfe_smote_balanced.csv")
