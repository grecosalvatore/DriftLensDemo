import time

from flask import Flask, render_template, request
from flask_socketio import SocketIO
from random import random
from threading import Lock
from datetime import datetime
from windows_manager.windows_generator import WindowsGenerator
from drift_lens.drift_lens import DriftLens
from utils import _utils
import h5py
import json
import os
"""
Background Thread
"""
thread = None
thread_lock = Lock()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins='*')

"""
Get current date time
"""
def get_current_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y %H:%M:%S")

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

"""
Generate random sequence of dummy sensor values and send it to our clients
"""
def background_thread():
    print("Generating random sensor values")
    while True:
        dummy_sensor_value = round(random() * 100, 3)
        socketio.emit('updateSensorData', {'value': dummy_sensor_value, "date": get_current_datetime()})
        socketio.sleep(1)

@app.route("/use_cases")
def use_cases():
    title = 'DriftLens'
    available_data = get_datasets_models_and_window_sizes()
    return render_template('use_cases.html', title=title, data=available_data)

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

def run_drift_detection_background_thread(form_parameters):
    print("\n\n")
    print("drift_lens_monitor Reciving")
    print("\n\n")
    selected_dataset = form_parameters['dataset']
    selected_model = form_parameters['model']
    selected_window_size = int(form_parameters['window_size'])
    selected_drift_pattern = form_parameters['drift_pattern']

    # Load Embedding
    new_unseen_embedding_path = f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}/saved_embeddings/new_unseen_embedding.hdf5"
    drifted_embedding_path = f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}/saved_embeddings/drifted_embedding.hdf5"

    E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(new_unseen_embedding_path)
    E_drift, Y_original_drift, Y_predicted_drift = load_embedding(drifted_embedding_path)

    training_label_list = [0, 1, 2]
    drift_label_list = [3]
    wg = WindowsGenerator(training_label_list, drift_label_list, E_new_unseen, Y_predicted_new_unseen,
                          Y_original_new_unseen, E_drift, Y_predicted_drift, Y_original_drift)

    # Create DriftLens Object
    dl = DriftLens(training_label_list)

    print("Loading Baseline")
    baseline = dl.load_baseline(folderpath=f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}/window_sizes/{selected_window_size}",
                                                  baseline_name="baseline")

    flag_shuffle = True
    flag_replacement = True

    #for i in range(1001):
    #    percentage = int((i / 1000) * 100)  # Calculate the percentage

        # Create a message with the current percentage
    #    message = json.dumps({"currentProgress": percentage})
    #    socketio.emit('UpdateProgressBarDataStreamGeneration', message)
    #    time.sleep(1)

    if selected_drift_pattern == "no_drift":
        print("no drift")
        selected_number_of_windows = int(form_parameters["number_of_windows_no_drift"])
        print(selected_number_of_windows)

        E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_without_drift_windows_generation(
            window_size=selected_window_size,
            n_windows=selected_number_of_windows,
            flag_shuffle=flag_shuffle,
            flag_replacement=flag_replacement,
            socketio=socketio)

    elif selected_drift_pattern == "sudden_drift":
        print("sudden drift")
        selected_number_of_windows = int(form_parameters["number_of_windows_sudden_drift"])
        selected_drift_offset = int(form_parameters["drift_offset_sudden_drift"])
        selected_drift_percentage = int(form_parameters["drift_percentage_sudden_drift"]) / 100
        print(selected_number_of_windows)
        print(selected_drift_offset)
        print(selected_drift_percentage)

        E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_incremental_drift_windows_generation(
            window_size=selected_window_size,
            n_windows=selected_number_of_windows,
            starting_drift_percentage=selected_drift_percentage,
            drift_increase_rate=0,
            drift_offset=selected_drift_offset,
            flag_shuffle=flag_shuffle,
            flag_replacement=flag_replacement,
            socketio=socketio)

    elif selected_drift_pattern == "incremental_drift":
        print("incremental drift")
        selected_number_of_windows = int(form_parameters["number_of_windows_incremental_drift"])
        selected_drift_offset = int(form_parameters["drift_offset_incremental_drift"])
        selected_starting_drift_percentage = int(form_parameters["drift_percentage_incremental_drift"]) / 100
        selected_increasing_drift_percentage = int(form_parameters["drift_increasing_percentage_incremental_drift"]) / 100

        E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_incremental_drift_windows_generation(
            window_size=selected_window_size,
            n_windows=selected_number_of_windows,
            starting_drift_percentage=selected_starting_drift_percentage,
            drift_increase_rate=selected_increasing_drift_percentage,
            drift_offset=selected_drift_offset,
            flag_shuffle=flag_shuffle,
            flag_replacement=flag_replacement,
            socketio=socketio)


    if selected_drift_pattern == "periodic_drift":
        print("periodic drift")

    for i, (E_w, y_pred, y_true) in enumerate(zip(E_windows, Y_predicted_windows, Y_original_windows)):
        window_distance = dl.compute_window_distribution_distances(E_w, y_pred)
        window_distance["window_id"] = i
        if isinstance(window_distance["batch"], complex):
            window_distance["batch"] = float(_utils.clear_complex_number(window_distance["batch"]).real)
        for l in training_label_list:
            if isinstance(window_distance["per-label"][str(l)], complex):
                print("clearing:", window_distance["per-label"][str(l)])
                window_distance["per-label"][str(l)] = float(_utils.clear_complex_number(window_distance["per-label"][str(l)]).real)
                print(window_distance["per-label"][str(l)])
        print(f"window: {i} - {window_distance}")

        per_label_distances = ",".join(str(v) for k,v in window_distance["per-label"].items())
        #yield f"data: {json.dumps(window_distance)}\n\n"
        print(window_distance["per-label"])
        socketio.emit('updateSensorData', {'batch_distance': window_distance["batch"], "per_label_distances":per_label_distances , "date": get_current_datetime()})
        socketio.sleep(0)

@app.route("/drift_lens_monitor", methods=["GET", "POST"])
def drift_lens_monitor():
    title = 'DriftLens'

    if request.method == "POST":
        all_parameters = request.form.to_dict()
    global thread
    print('Client connected')

    global thread
    with thread_lock:
        if thread is None:
            #thread = socketio.start_background_task(background_thread)
            thread = socketio.start_background_task(run_drift_detection_background_thread, all_parameters)

    return render_template('drift_lens_monitor.html', title=title, num_labels=3, label_names=",".join(["label1", "label2", "label3"]))


"""
Serve root index file
"""
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_use_case')
def upload_use_case():
    return render_template('upload_use_case.html')
"""
Decorator for connect
"""
"""@socketio.on('connect')
def connect():
    global thread
    print('Client connected')

    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
"""
"""
Decorator for disconnect
"""
@socketio.on('disconnect')
def disconnect():
    print('Client disconnected',  request.sid)

if __name__ == '__main__':
    socketio.run(app)