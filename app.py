import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, render_template, request, redirect, url_for, make_response, Response
import plotly.express as px
import time
import webbrowser
import h5py
import numpy as np
from windows_manager.windows_generator import WindowsGenerator
import json
import random

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
def drift_lens_monitor():
    # ... (your existing code)

    form_parameters_str = request.args.get("form_params")
    print("Drift Lens Monitor: ", form_parameters_str)
    print(type(form_parameters_str))
    # Replace single quotes with double quotes
    form_parameters_str = form_parameters_str.replace("'", "\"")
    form_parameters = json.loads(form_parameters_str)
    time.sleep(20)
    selected_dataset = form_parameters['dataset']
    selected_model = form_parameters['model']
    selected_window_size = int(form_parameters['window_size'])
    selected_drift_pattern = form_parameters['drift_pattern']


    def generate_chart_updates():
        while True:
            # Replace this with your logic to generate new data for the chart
            new_data = [random.randint(1, 10) for _ in range(10)]
            yield f"data: {json.dumps(new_data)}\n\n"
            time.sleep(1)  # Adjust the sleep time as needed

    return Response(generate_chart_updates(), content_type='text/event-stream')



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
        selected_window_size = int(request.form.get("window_size"))
        selected_drift_pattern = request.form.get("drift_pattern")
        all_parameters = request.form.to_dict()

        # Load Embedding
        new_unseen_embedding_path = f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}/saved_embeddings/new_unseen_embedding.hdf5"
        drifted_embedding_path = f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}/saved_embeddings/drifted_embedding.hdf5"

        E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(new_unseen_embedding_path)
        E_drift, Y_original_drift, Y_predicted_drift = load_embedding(drifted_embedding_path)

        training_label_list = [0, 1, 2]
        drift_label_list = [3]
        wg = WindowsGenerator(training_label_list, drift_label_list, E_new_unseen, Y_predicted_new_unseen,
                                                Y_original_new_unseen, E_drift, Y_predicted_drift, Y_original_drift)

        flag_shuffle = True
        flag_replacement = True

        if selected_drift_pattern == "no_drift":
            print("no drift")
            selected_number_of_windows = int(request.form.get("number_of_windows_no_drift"))
            print(selected_number_of_windows)

            E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_without_drift_windows_generation(window_size=selected_window_size,
                                                                                                              n_windows=selected_number_of_windows,
                                                                                                              flag_shuffle=flag_shuffle,
                                                                                                              flag_replacement=flag_replacement)

        elif selected_drift_pattern == "sudden_drift":
            print("sudden drift")
            selected_number_of_windows = int(request.form.get("number_of_windows_sudden_drift"))
            selected_drift_offset = int(request.form.get("drift_offset_sudden_drift"))
            selected_drift_percentage = int(request.form.get("drift_percentage_sudden_drift"))/100
            print(selected_number_of_windows)
            print(selected_drift_offset)
            print(selected_drift_percentage)

            E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_incremental_drift_windows_generation(window_size=selected_window_size,
                                                                                                                  n_windows=selected_number_of_windows,
                                                                                                                  starting_drift_percentage=selected_drift_percentage,
                                                                                                                  drift_increase_rate=0,
                                                                                                                  drift_offset=selected_drift_offset,
                                                                                                                  flag_shuffle=flag_shuffle,
                                                                                                                  flag_replacement=flag_replacement)
            for i,y in enumerate(Y_original_windows):
                count_n =  np.sum(y == 3)
                print(f"window: {i} - {count_n}")

        elif selected_drift_pattern == "incremental_drift":
            print("incremental drift")
            selected_number_of_windows = int(request.form.get("number_of_windows_incremental_drift"))
            selected_drift_offset = int(request.form.get("drift_offset_incremental_drift"))
            selected_starting_drift_percentage = int(request.form.get("drift_percentage_incremental_drift"))/100
            selected_increasing_drift_percentage = int(request.form.get("drift_increasing_percentage_incremental_drift"))/100

            print(selected_starting_drift_percentage)
            print(selected_increasing_drift_percentage)

            E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_incremental_drift_windows_generation(window_size=selected_window_size,
                                                                                                                  n_windows=selected_number_of_windows,
                                                                                                                  starting_drift_percentage=selected_starting_drift_percentage,
                                                                                                                  drift_increase_rate=selected_increasing_drift_percentage,
                                                                                                                  drift_offset=selected_drift_offset,
                                                                                                                  flag_shuffle=flag_shuffle,
                                                                                                                  flag_replacement=flag_replacement)

            for i,y in enumerate(Y_original_windows):
                count_n =  np.sum(y == 3)
                print(f"window: {i} - {count_n}")

        if selected_drift_pattern == "periodic_drift":
            print("periodic drift")


        #for E_win in E_windows:
            # Do something and update plot

        # JavaScript to open a new window and redirect the current window
        script = """
        <script>
            if (!window.alreadyOpened) {
                // Open a new window with drift_lens_monitor
                var newWindow = window.open("/drift_lens_monitor?form_params=%s", "_blank");
                newWindow.focus();

                // Delay the redirection by a few milliseconds to ensure the new window has opened
                setTimeout(function() {
                    // Redirect the current window to /run_experiment
                    window.location.href = "/run_experiment";
                }, 100); // You can adjust the delay in milliseconds if needed

                window.alreadyOpened = true; // Set a flag to indicate that the new window is opened
            }
        </script>
        """ % (all_parameters)

        response = make_response(
            f"dataset={selected_dataset}, Selected Model: {selected_model}, Selected Window Size: {selected_window_size}, Selected Drift Pattern: {selected_drift_pattern} {script}")
    else:
        # This block should execute when you access /run_experiment directly
        response = make_response(render_template("run_experiment.html", dataset=selected_dataset, model=selected_model, drift_pattern=selected_drift_pattern))

    return response



if __name__ == '__main__':
    app.run(debug=True)