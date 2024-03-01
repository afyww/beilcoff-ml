from flask import Flask, render_template, request, redirect
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import base64
import io
import os

import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def perform_association_rule_analysis(file_path):
    data = pd.read_excel(file_path)
    data['Pada Tanggal'] = pd.to_datetime(data['Pada Tanggal'])
    data['Menu'] = data['Menu'].str.split(', ')
    data = data.explode('Menu')
    data['Menu'] = data['Menu'].str.lower()

    basket = (data.groupby(['No', 'Menu'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('No'))

    def hot_encode(x):
        if x <= 0:
            return 0
        if x >= 0:
            return 1

    basket_encode = basket.applymap(hot_encode)
    basket = basket_encode

    frq_items = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frq_items, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

    # Generate heatmap plot
    plt.figure(figsize=(10, 6))
    heatmap_data = rules.pivot_table(index='antecedents', columns='consequents', values='lift')
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
    plt.title('Heatmap of Lift Values for Association Rules')

    # Save the plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    plt.close()  # Close the plot to avoid issues with multiple plots in a single process

    # Convert the image buffer to base64 for embedding in HTML
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    return rules, img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        rules, img_base64 = perform_association_rule_analysis(file_path)

        # Convert DataFrame to HTML
        rules_html = rules.to_html(classes="table table-striped", index=False)

        return render_template('index.html', rules_html=rules_html, img_base64=img_base64)

    return render_template('index.html', rules_html=None, img_base64=None)

if __name__ == '__main__':
    app.run(debug=True)
