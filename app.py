from flask import Flask, render_template, request, redirect, url_for, make_response
import plotly.express as px
import os
import time
import webbrowser

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


@app.route("/run_experiment", methods=["GET", "POST"])
def run_experiment():
    selected_dataset = ""
    selected_model = ""
    selected_window_size = ""
    selected_drift_pattern = ""
    if request.method == "POST":
        # This block should execute only when the form is submitted via POST
        selected_dataset = request.form.get("dataset")
        selected_model = request.form.get("model")
        selected_window_size = request.form.get("window_size")
        selected_drift_pattern = request.form.get("drift_pattern")

        if selected_drift_pattern == "no_drift":
            print("no drift")
        elif selected_drift_pattern == "sudden_drift":
            print("sudden drift")
        elif selected_drift_pattern == "incremental_drift":
            print("incremental drift")
        if selected_drift_pattern == "periodic_drift":
            print("periodic drift")

        # JavaScript to open a new window and redirect the current window
        # JavaScript to open a new window and redirect the current window
        script = """
        <script>
            if (!window.alreadyOpened) {
                // Open a new window with drift_lens_monitor
                var newWindow = window.open("/drift_lens_monitor", "_blank");
                newWindow.focus();

                // Delay the redirection by a few milliseconds to ensure the new window has opened
                setTimeout(function() {
                    // Redirect the current window to /run_experiment with query parameters
                    var dataset = "%s";  // Note the use of placeholders here
                    var model = "%s";
                    window.location.href = "/run_experiment?dataset=" + dataset + "&model=" + model;
                }, 100); // You can adjust the delay in milliseconds if needed

                window.alreadyOpened = true; // Set a flag to indicate that the new window is opened
            }
        </script>
        """ % (selected_dataset, selected_model)

        response = make_response(
            f"dataset={selected_dataset}, Selected Model: {selected_model}, Selected Window Size: {selected_window_size}, Selected Drift Pattern: {selected_drift_pattern} {script}")
    else:
        # This block should execute when you access /run_experiment directly
        response = make_response(render_template("run_experiment.html", dataset=selected_dataset, model=selected_model, drift_pattern=selected_drift_pattern))

    return response



if __name__ == '__main__':
    app.run(debug=True)
