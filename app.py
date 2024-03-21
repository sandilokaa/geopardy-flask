from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

#---- Import the Utils ----#

with open('./utils/accuracy.pkl', 'rb') as file:
    accuracy_ensemble = pickle.load(file)

with open('./utils/logreg_model.pkl', 'rb') as file:
    logreg_model = pickle.load(file)

with open('./utils/gb_model.pkl', 'rb') as file:
    gb_model = pickle.load(file)

#---- End Import the Utils ----#
    

#---- Define Endpoint ----#

@app.route('/accuracy', methods=['GET'])
def rainfall_accuracy():
    return jsonify({
        'accuracy_ensemble': [
            {
                'accuracy': accuracy_ensemble
            }
        ]
    })


@app.route('/predict', methods=['POST'])
def rainfall_predict():

    data = request.json
    new_data = pd.DataFrame(data, index=[0])

    ensemble_prob_new = (logreg_model.predict_proba(new_data)[:, 1] + gb_model.predict_proba(new_data)[:, 1]) / 2
    ensemble_pred_new = (ensemble_prob_new > 0.5).astype(int)

    prediction_result = {
        'Probability': float(ensemble_prob_new),
        'Prediction': 'Yes, it will rain.' if ensemble_pred_new > 0.5 else 'No, it will not rain.'
    }

    return jsonify(prediction_result)


#---- Define Endpoint ----#

if __name__ == "__main__":
    app.run(debug=True)