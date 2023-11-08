from flask import Flask, render_template, request, redirect, url_for

import plotly.express as px
import os

app = Flask(__name__)

@app.route("/")
def home():
    title = 'DriftLens'
    return render_template('home.html', title=title)

@app.route("/use_cases")
def use_cases():
    title = 'DriftLens'
    available_datasets = get_datasets_and_models()
    return render_template('use_cases.html', title=title, datasets=available_datasets)

@app.route("/upload")
def upload():
    title = 'DriftLens'
    return render_template('upload.html', title=title)

@app.route("/drift_lens_monitor")
def plotly_chart():
    # Create or retrieve data for your Plotly chart
    data = [1, 2, 3, 4, 5]

    # Create the Plotly chart
    fig = px.line(x=list(range(1, len(data) + 1)), y=data, title='Your Plotly Chart')

    # Generate HTML for the Plotly chart
    plotly_chart_div = fig.to_html(full_html=False)

    return render_template('drift_lens_monitor.html', plotly_chart=plotly_chart_div)



def get_datasets_and_models():
    base_directory = "use_cases/datasets"
    datasets = []
    for dataset_dir in os.listdir(base_directory):
        if os.path.isdir(os.path.join(base_directory, dataset_dir)):
            models = []
            models_path = os.path.join(base_directory, dataset_dir, "models")
            if os.path.exists(models_path):
                for model_dir in os.listdir(models_path):
                    if os.path.isdir(os.path.join(models_path, model_dir)):
                        models.append(model_dir)
            datasets.append({"name": dataset_dir, "models": models})
    return datasets

@app.route("/process_selection", methods=["POST"])
def process_selection():
    selected_dataset = request.form.get("dataset")
    selected_model = request.form.get("model")

    # You can use the selected_dataset and selected_model values as needed.
    # For example, you can pass them to a different template to display the results.

    return f"Selected Dataset: {selected_dataset}, Selected Model: {selected_model}"


if __name__ == '__main__':
    app.run(debug=True)
