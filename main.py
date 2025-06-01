from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model_data = joblib.load("stacking_classifier_model.pkl")
model = model_data["model"]
encoders = model_data["encoders"]
all_features = model_data["features"]  # full list including target

# Remove the target feature 'Depression' from input features
features = [f for f in all_features if f.lower() != "depression"]


@app.route('/')
def home():
    return render_template('index.html', features=features)


@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    for feature in features:
        value = request.form.get(feature)
        if value is None or value.strip() == "":
            return render_template('index.html', prediction_text=f"Missing value for {feature}", features=features)

        if feature in encoders:
            try:
                # Encode categorical feature
                value = encoders[feature].transform([value])[0]
            except Exception as e:
                return render_template('index.html', prediction_text=f"Encoding error for {feature}: {e}", features=features)
        else:
            try:
                # Convert numerical feature
                value = float(value)
            except ValueError:
                return render_template('index.html', prediction_text=f"Invalid input for {feature}. Must be a number.", features=features)

        input_data.append(value)

    df_input = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(df_input)[0]

    if prediction == 1:
        message = (
            "Your results suggest you might be prone to depression. "
            "Remember, recognizing this is the first step towards healing. "
            "You are not alone, and with support and care, brighter days are ahead."
        )
    else:
        message = (
            "Great news! You appear to be resilient against depression at this time. "
            "Keep nurturing your mental well-being and stay positive—you’re doing well!"
        )
    
    return render_template('index.html',
                           prediction_text=message,
                           features=features)



if __name__ == '__main__':
    app.run(debug=True)
