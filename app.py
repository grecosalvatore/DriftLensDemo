from flask import Flask, render_template, request, redirect, url_for, make_response
import plotly.express as px
import os
import time
import webbrowser
import h5py

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

def load_embedding(filepath, E_name=None, Y_original_name=None, Y_predicted_name=None):
    if filepath is not None:
        with h5py.File(filepath, "r") as hf:
            if E_name is None:
                E = hf["E"][()]
            else:
                E = hf[E_name][()]
            if Y_original_name is None:
                Y_original = hf["Y_original"][()]
            else:
                Y_original = hf[Y_original_name][()]
            if Y_predicted_name is None:
                Y_predicted = hf["Y_predicted"][()]
            else:
                Y_predicted = hf[Y_predicted_name][()]
    else:
        raise Exception("Experiment Manager: Error in loading the embedding file. Please set the embedding paths in the configuration file.")
    return E, Y_original, Y_predicted

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

        # Load Embedding
        new_unseen_embedding_path = f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}/saved_embeddings/new_unseen_embedding.hdf5"
        drifted_embedding_path = f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}/saved_embeddings/drifted_embedding.hdf5"


        if selected_drift_pattern == "no_drift":
            print("no drift")
            selected_number_of_windows = request.form.get("number_of_windows_no_drift")
            print(selected_number_of_windows)
            E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(new_unseen_embedding_path)
        elif selected_drift_pattern == "sudden_drift":
            print("sudden drift")
            selected_number_of_windows = request.form.get("number_of_windows_sudden_drift")
            selected_drift_offset = request.form.get("drift_offset_sudden_drift")
            selected_drift_percentage = request.form.get("drift_percentage_sudden_drift")
            print(selected_number_of_windows)
            print(selected_drift_offset)
            print(selected_drift_percentage)
            E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(new_unseen_embedding_path)
            E_drift, Y_original_drift, Y_predicted_drift = load_embedding(drifted_embedding_path)

        elif selected_drift_pattern == "incremental_drift":
            print("incremental drift")
            selected_number_of_windows = request.form.get("number_of_windows_incremental_drift")
            selected_drift_offset = request.form.get("drift_offset_incremental_drift")
            selected_starting_drift_percentage = request.form.get("drift_percentage_incremental_drift")
            selected_increasing_drift_percentage = request.form.get("drift_increasing_percentage_incremental_drift")
            print(selected_number_of_windows)
            print(selected_drift_offset)
            print(selected_starting_drift_percentage)
            print(selected_increasing_drift_percentage)

            E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(new_unseen_embedding_path)
            E_drift, Y_original_drift, Y_predicted_drift = load_embedding(drifted_embedding_path)
        if selected_drift_pattern == "periodic_drift":
            print("periodic drift")

            E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(new_unseen_embedding_path)
            E_drift, Y_original_drift, Y_predicted_drift = load_embedding(drifted_embedding_path)


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
