from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import gdown

# URL of the model file on Google Drive
url = 'https://cyberspaceinfrastructure-my.sharepoint.com/:u:/g/personal/amit_mydataup_onmicrosoft_com/EUwBA9E2-ONPpguKG49UMkIBXM37S-4fUOOMzqFcRa8nLA?e=sz7ehP'
output = 'best_rf_model.joblib'

# Function to download the model file from cloud storage
def download_model():
    print("Downloading model...")
    gdown.download(url, output, quiet=False)  # This downloads the model to your local system
    print("Download complete.")

# Download the model and load it
download_model()
model = load(output)
# Download the model



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
