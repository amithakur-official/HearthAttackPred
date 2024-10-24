from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the saved Random Forest model
model = load('best_rf_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # Return the prediction and probability as a JSON response
    return jsonify({'prediction': int(prediction[0]), 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)
