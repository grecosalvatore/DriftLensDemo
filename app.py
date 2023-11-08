from flask import Flask, render_template, request, redirect, url_for
import plotly.express as px
import os

app = Flask(__name__)

def get_datasets_models_and_window_sizes():
    base_directory = "static/use_cases/datasets"
    datasets = []

    for dataset_dir in os.listdir(base_directory):
        if os.path.isdir(os.path.join(base_directory, dataset_dir)):
            models = []

            dataset_path = os.path.join(base_directory, dataset_dir)
            models_path = os.path.join(dataset_path, "models")

            if os.path.exists(models_path):
                for model_dir in os.listdir(models_path):
                    if os.path.isdir(os.path.join(models_path, model_dir)):
                        window_sizes = []

                        model_path = os.path.join(models_path, model_dir)
                        window_sizes_path = os.path.join(model_path, "window_sizes")

                        if os.path.exists(window_sizes_path):
                            window_sizes = [ws for ws in os.listdir(window_sizes_path) if os.path.isdir(os.path.join(window_sizes_path, ws))]

                        models.append({"name": model_dir, "window_sizes": window_sizes})

            datasets.append({"name": dataset_dir, "models": models})

    return datasets

@app.route("/")
def home():
    title = 'DriftLens'
    return render_template('home.html', title=title)

@app.route("/use_cases")
def use_cases():
    title = 'DriftLens'
    available_data = get_datasets_models_and_window_sizes()
    return render_template('use_cases.html', title=title, data=available_data)

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

@app.route("/process_selection", methods=["POST"])
def process_selection():
    selected_dataset = request.form.get("dataset")
    selected_model = request.form.get("model")
    selected_window_size = request.form.get("window_size")

    # You can use the selected_dataset, selected_model, and selected_window_size values as needed.
    # For example, you can pass them to a different template to display the results.

    return f"Selected Dataset: {selected_dataset}, Selected Model: {selected_model}, Selected Window Size: {selected_window_size}"

if __name__ == '__main__':
    app.run(debug=True)
