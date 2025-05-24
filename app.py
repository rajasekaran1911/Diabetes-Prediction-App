from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("diabetes_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c'])
        hypertension = int(request.form['hypertension'])
        family_history = request.form['family_history']
        diet_quality = request.form['diet_quality']

        # Encode categorical inputs safely
        def safe_encode(label, encoder):
            if label in encoder.classes_:
                return encoder.transform([label])[0]
            else:
                return encoder.transform([encoder.classes_[0]])[0]  # Default to first class

        family_history_encoded = safe_encode(family_history, label_encoders["FamilyHistoryDiabetes"])
        diet_quality_encoded = safe_encode(diet_quality, label_encoders["DietQuality"])

        # Prepare input for model
        features = np.array([[age, bmi, hba1c, hypertension, family_history_encoded, diet_quality_encoded]])

        # Make prediction
        prediction = model.predict(features)

        # Convert numerical prediction to "Yes" or "No"
        diagnosis = "Yes" if prediction[0] == 1 else "No"

        return render_template('result.html', diagnosis=diagnosis)

    except Exception as e:
        return render_template('result.html', diagnosis=f"⚠️ Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
