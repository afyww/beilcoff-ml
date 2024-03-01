from flask import Flask, request, jsonify, after_this_request
from flask_cors import CORS
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from io import BytesIO
import base64
import os
import time
import json

app = Flask(__name__)
CORS(app)

def generate_plot(ts, forecast_values):
    plt.figure(figsize=(10, 6))
    plt.plot(ts.index, ts.values, label='Observasi')
    plt.plot(forecast_values.index, forecast_values.values, color='red', label='Prediksi')
    plt.title('Prediksi - Satu Minggu ke Depan')
    plt.xlabel('Tanggal')
    plt.ylabel('Total Amount')
    plt.legend()

    # Format x-axis labels to show only month and day
    plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d'))

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_filename = f"static/images/forecast_{int(time.time())}.png"

    # Create the 'static/images' directory if it does not exist
    os.makedirs(os.path.dirname(image_filename), exist_ok=True)

    # Save the plot
    plt.savefig(image_filename, format='png')
    plt.close()

    return image_filename

@app.route("/api", methods=['POST'])
def api():
    if 'file' not in request.files:
        return jsonify({"error": "File not found"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if file:
        data = pd.read_excel(file, parse_dates=['Pada Tanggal'])
        data.set_index('Pada Tanggal', inplace=True)
        ts = data['Total Amount']

        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)

        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
        result = model.fit()

        forecast_steps = 7
        forecast_index = pd.date_range(ts.index[-1], periods=forecast_steps + 1, freq='D')[1:]
        forecast = result.get_forecast(steps=forecast_steps, index=forecast_index)
        forecast_values = forecast.predicted_mean

        @after_this_request
        def add_image(response):
            image_filename = None

            try:
                image_filename = generate_plot(ts, forecast_values)
            except Exception as e:
                print(f"Error generating plot: {e}")

            if image_filename:
                response.data = json.dumps({"image_filename": image_filename})
                response.headers['Content-Type'] = 'application/json'

            return response

        return "Processing..."

if __name__ == "__main__":
    app.run(debug=True)
