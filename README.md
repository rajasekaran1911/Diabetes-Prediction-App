🩺 Diabetes Prediction Web Application
A web-based machine learning application to predict the likelihood of diabetes based on user health inputs such as Age, BMI, HbA1c levels, Hypertension, Family History of Diabetes, and Diet Quality.

🔍 Problem Statement
Diabetes is a chronic condition that affects millions globally. Early detection is crucial for effective prevention and treatment. This project aims to provide a simple, interactive, and accurate tool for predicting the risk of diabetes using machine learning.

🎯 Scope
This project demonstrates how machine learning models can be integrated into a web application to provide real-time health predictions. It includes model training, evaluation, user interface design, and result visualization.

🛠️ Technologies Used
Python 3

Flask – Web framework

scikit-learn – ML model training

Pandas, NumPy – Data processing

Matplotlib, Seaborn – Visualization

HTML, CSS – Frontend design

Joblib – Model persistence

📁 Project Structure
php
Copy
Edit
DiabetesPredictionApp/
│
├── static/                     # CSS & image files
│   └── style.css
│
├── templates/                  # HTML templates
│   ├── base.html               # Common layout with sidebar
│   ├── index.html              # Input form
│   ├── result.html             # Prediction result page
│   ├── accuracy.html           # Model accuracy visualization
│   └── analytics.html          # Feature importance & analytics
│
├── final_diabetics_data.csv    # Input dataset
├── app.py                      # Flask backend
├── model.py                    # ML model training and saving
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
📊 Features
✔️ Predicts diabetes diagnosis (Yes/No) based on health input

✔️ Dashboard with sidebar navigation

✔️ Real-time prediction and display

✔️ Model performance (Accuracy, Confusion Matrix)

✔️ Analytics and Feature Importance Graphs

✔️ Responsive and styled UI with custom CSS
