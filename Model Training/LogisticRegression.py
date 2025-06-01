import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("C:/School Files/Depression Predictor Model/Model Testing/annova_smote_balanced.csv")  # Replace with your actual file path

# Define features and target
features = ["Age", 'City','Profession',"Academic Pressure", "CGPA", "Study Satisfaction", "Sleep Duration", "Dietary Habits","Degree",
            "Suicidal thoughts ?", "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"]
X = df[features]
y = df["Depression"]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Encode target if it's not numeric
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, solver='liblinear')  # Use 'liblinear' for binary
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("📊 Logistic Regression Evaluation in Annova Feature Selection:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
