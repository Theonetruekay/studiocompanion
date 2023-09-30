from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import requests

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol')

    # Fetch data from Alpha Vantage
    API_KEY = 'your_alpha_vantage_api_key_here'
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()['Time Series (5min)']

    # Prepare data
    df = pd.DataFrame.from_dict(data).T['1. open'].astype(float).reset_index()
    df['timestamp'] = pd.to_datetime(df['index'])
    df = df[['timestamp', '1. open']]

    # Train model
    model = LinearRegression()
    model.fit(np.array(range(len(df))).reshape(-1, 1), df['1. open'])

    # Predict next value
    next_val = model.predict(np.array([[len(df)]]))

    return jsonify({'prediction': float(next_val)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
