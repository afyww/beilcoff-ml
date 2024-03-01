from flask import Flask, request, jsonify
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
from mlxtend.frequent_patterns import apriori, association_rules

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

def hot_encode(x):
    return 1 if x >= 0 else 0

def perform_association_rule_analysis(file_path):
    with pd.ExcelFile(file_path) as xls:
        data = pd.read_excel(xls, parse_dates=['Pada Tanggal'])
        
    data['Menu'] = data['Menu'].str.lower()
    basket = (data.groupby(['No', 'Menu'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('No'))

    def hot_encode(x):
        return 1 if x >= 0 else 0

    basket_encode = basket.applymap(hot_encode)
    basket = basket_encode

    frq_items = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frq_items, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

    # Generate heatmap plot
    plt.figure(figsize=(10, 6))
    heatmap_data = rules.pivot_table(index='antecedents', columns='consequents', values='lift')
    plt.pcolor(heatmap_data, cmap='coolwarm')
    plt.colorbar()
    plt.title('Heatmap of Lift Values for Association Rules')

    # Save the plot to a BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    plt.close()  # Close the plot to avoid issues with multiple plots in a single process

    # Convert the image buffer to base64 for embedding in HTML
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    # Convert frozenset values in rules to lists
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))

    return rules, img_base64


@app.route("/api", methods=['POST'])
def api():
    if 'file' not in request.files:
        return jsonify({"error": "File not found"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if file:
        data = pd.read_excel(file, parse_dates=['Pada Tanggal'], index_col='Pada Tanggal')
        ts = data['Total Amount']

        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)

        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
        result = model.fit()

        forecast_steps = 7
        forecast_index = pd.date_range(ts.index[-1], periods=forecast_steps + 1, freq='D')[1:]
        forecast = result.get_forecast(steps=forecast_steps, index=forecast_index)
        forecast_values = forecast.predicted_mean

        try:
            image_filename = generate_plot(ts, forecast_values)
        except Exception as e:
            print(f"Error generating plot: {e}")
            image_filename = None

        data_rules, img_base64 = perform_association_rule_analysis(file)

        data_rules_list = data_rules.to_dict(orient='records')
        response_data = {
            "image_filename": image_filename,
            "association_rules": data_rules_list,
            "heatmap_image_base64": img_base64
        }

        return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)
