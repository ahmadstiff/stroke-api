from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

CORS(app)

model_filename = 'stroke_model.pkl'
model_data = joblib.load(model_filename)
model = model_data['model']
scaler = model_data['scaler']

feature_columns = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'gender_Other', 'ever_married_Yes',
    'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
    'work_type_children', 'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not all(feature in data for feature in feature_columns):
        return jsonify({'error': 'Missing required features in input'}), 400

    input_data = pd.DataFrame([data], columns=feature_columns)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    return jsonify({'stroke_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
