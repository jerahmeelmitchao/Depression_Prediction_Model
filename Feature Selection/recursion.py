import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("C:/School Files/Depression Predictor Model/Datasets/dataset_standardization.csv")

# Split into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode target if it's categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize model
model = LogisticRegression(max_iter=1000, solver='liblinear')

# RFECV for automatic feature selection with cross-validation
rfecv = RFECV(
    estimator=model,
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy'  # You can change to 'f1', 'roc_auc', etc.
)

# Fit RFECV
rfecv.fit(X, y_encoded)

# Get selected features mask
feature_mask = rfecv.support_

# Print optimal number of features
print("✅ Optimal number of features:", rfecv.n_features_)

# Print list of all features with check or cross
print("\nFeature Selection Summary:")
for feature, selected in zip(X.columns, feature_mask):
    status = "✅ Selected" if selected else "❌ Not Selected"
    print(f"{feature}: {status}")

# Plot feature selection results
plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (accuracy)")
plt.plot(
    range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
    rfecv.cv_results_['mean_test_score'],
    marker='o'
)
plt.title("Feature Selection using RFECV")
plt.grid(True)
plt.tight_layout()
plt.show()
