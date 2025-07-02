from flask import Flask, request, render_template
import numpy as np
import pickle
import os
import gdown

app = Flask(__name__)

# Function to download files from Google Drive
def download_file(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists.")

# Google Drive file IDs (replace with your own)
MODEL_FILE_ID = '1DzOfcGVu3Mx96DZ_1f3igYeKs_Ip3K7d'
SCALER_FILE_ID = '1FFSW-aYqOpq9HbKl_cX8-w15qFEy-LZO'

# Download model and scaler if not present
download_file(MODEL_FILE_ID, 'house_model.pkl')
download_file(SCALER_FILE_ID, 'scaler.pkl')

# Load model and scaler
model = pickle.load(open('house_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction[0]*100000:.2f}')

if __name__ == '__main__':
    app.run(debug=False)
