import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("C:/School Files/Depression Predictor Model/Model Testing/stackfeautureremoval_withsmote.csv")  

# Define features and target
features = ['Age', 'City', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration',
            'Dietary Habits', 'Degree', 'Suicidal thoughts', 'Work/Study Hours', 'Financial Stress',
            'Family History of Mental Illness', 'Profession', 'Gender']
X = df[features].copy()
y = df['Depression']

# Encode categorical columns and store encoders
encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define base and final estimators
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('nb', GaussianNB())
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=42)
)

# Train model
stacking_clf.fit(X_train, y_train)

# Evaluate
y_pred_stack = stacking_clf.predict(X_test)
print("=== Stacking Classifier Performance ===")
print(classification_report(y_test, y_pred_stack))
print(f"Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_stack):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_stack):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_stack):.4f}")

# Save model and encoders
joblib.dump({
    "model": stacking_clf,
    "encoders": encoders,
    "features": features
}, "C:/School Files/Depression Predictor Model/stacking_classifier_model.pkl")
