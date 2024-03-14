from flask import Flask, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

with open('./utils/accuracy.pkl', 'rb') as file:
    accuracy_ensemble = pickle.load(file)

with open('./utils/logreg_model.pkl', 'rb') as file:
    logreg_model = pickle.load(file)

with open('./utils/gb_model.pkl', 'rb') as file:
    gb_model = pickle.load(file)

@app.route('/accuracy', methods=['GET'])
def rainfall_accuracy():
    return jsonify({
        'accuracy_ensemble': [
            {
                'accuracy': accuracy_ensemble
            }
        ]
    })


@app.route('/predict', methods=['GET'])
def rainfall_predict():

    new_data = pd.DataFrame({
        'MinTemp': 20.2,
        'MaxTemp': 32.3,
        'Rainfall': 5.2, 
        'Evaporation': 12.3,
        'Sunshine': 12.8, 
        'WindGustSpeed': 56.3, 
        'WindSpeed9am': 45.3, 
        'WindSpeed3pm': 32.4, 
        'Humidity9am': 43.6, 
        'Humidity3pm': 23.5, 
        'Pressure9am': 1700.43, 
        'Pressure3pm': 1023.4, 
        'Cloud9am': 72.3, 
        'Cloud3pm': 51.3, 
        'Temp3pm': 42.2,
        'RainToday': 1
    }, [0])

    ensemble_prob_new = (logreg_model.predict_proba(new_data)[:, 1] + gb_model.predict_proba(new_data)[:, 1]) / 2
    ensemble_pred_new = (ensemble_prob_new > 0.5).astype(int)

    prediction_result = {
        'Probability': float(ensemble_prob_new),
        'Prediction': 'Yes, it will rain.' if ensemble_pred_new == 1 else 'No, it will not rain.'
    }

    return jsonify(prediction_result)

if __name__ == "__main__":
    app.run(debug=True)