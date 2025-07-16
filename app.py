from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app) #this will allow cors requests
model = load_model('lstm_7day_model.keras',compile=False)
scaler = joblib.load('scaler.save')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.json['ticker']
    data = yf.download(ticker, period='3y')
    close = data[['Close']]
    
    scaled = scaler.transform(close)
    last_60 = scaled[-60:]
    input_data = np.array([last_60])
    pred_scaled = model.predict(input_data)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    return jsonify({'predictions': pred.tolist()})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT env var
    app.run(host='0.0.0.0', port=port)