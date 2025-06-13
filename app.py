from flask import Flask, request, render_template
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('house_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

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
    app.run(debug=True)
