import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from Excel file
df = pd.read_csv("C:/School Files/Depression Predictor Model/Model Testing/rfe_smote_balanced.csv")  # Replace with your actual file path

# Features and target
features = ["Suicidal thoughts ?", "Academic Pressure","CGPA","Study Satisfaction","Sleep Duration","Work/Study Hours", "Financial Stress", "Age","City","Degree"]
X = df[features]
y = df["Depression"]

# Optional: encode categorical features if needed
from sklearn.preprocessing import LabelEncoder
X = X.apply(LabelEncoder().fit_transform)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize and train Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred = nb_model.predict(X_test)

# Evaluation
print("📊 Naive Bayes Evaluation in Recursive Selection:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()
