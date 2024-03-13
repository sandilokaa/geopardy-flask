from flask import Flask, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

with open('accuracy.pkl', 'rb') as file:
    accuracy_ensemble = pickle.load(file)

@app.route("/accuracy")
def rainfall_prediction():
    return jsonify({
        'accuracy_ensemble': {
            'accuracy': accuracy_ensemble
        }
    })

if __name__ == "__main__":
    app.run(debug=True)