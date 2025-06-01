import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway
from imblearn.over_sampling import SMOTE  # ✅ NEW

# Load dataset
data = pd.read_csv("C:/School Files/Depression Predictor Model/Datasets/dataset_standardization.csv")

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Drop 'id' column if exists
if 'id' in X.columns:
    X = X.drop(columns=['id'])

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------------------------
# 1. ANOVA Feature Selection
# -------------------------
anova_features = []
for col in X.columns:
    groups = [X[col][y_encoded == label] for label in np.unique(y_encoded)]
    try:
        f_stat, p_val = f_oneway(*groups)
        if p_val < 0.05:
            anova_features.append(col)
    except:
        continue

# -------------------------
# 2. RFECV Recursive Feature Selection
# -------------------------
model = LogisticRegression(max_iter=1000, solver='liblinear')
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X, y_encoded)
rfecv_features = X.columns[rfecv.support_].tolist()

# -------------------------
# 3. Random Forest Importance
# -------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y_encoded)
rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
rf_features = rf_importances[rf_importances['Importance'] > 0.01]['Feature'].tolist()

# -------------------------
# 4. Combine all selections
# -------------------------
all_selected = pd.Series(anova_features + rfecv_features + rf_features)
final_features = all_selected.value_counts()
final_selected_features = final_features[final_features >= 2].index.tolist()

# -------------------------
# 5. Apply SMOTE AFTER Feature Selection
# -------------------------
X_final = X[final_selected_features]
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_final, y_encoded)

# -------------------------
# 6. Save cleaned and balanced dataset
# -------------------------
resampled_data = pd.DataFrame(X_resampled, columns=final_selected_features)
resampled_data['Depression'] = y_resampled
resampled_data.to_csv("C:/School Files/Depression Predictor Model/Model Testing/stackfeautureremoval_withsmote.csv", index=False)

# -------------------------
# 7. Display Info
# -------------------------
print("✅ Final Selected Features (appeared in ≥ 2 methods):")
print(final_selected_features)

print("\n❌ Features Dropped:")
features_to_drop = [col for col in X.columns if col not in final_selected_features]
print(features_to_drop)

print(f"\n✅ Dataset with SMOTE saved. Total samples: {len(resampled_data)}")
