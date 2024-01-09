import time
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from random import random
from threading import Lock
from datetime import datetime
from windows_manager.windows_generator import WindowsGenerator
from drift_lens.drift_lens import DriftLens
import drift_lens._baseline as _baseline
from utils import _utils
import h5py
import json
import os
import yaml
import shutil


""" Background Thread """
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
    #return now.strftime("%m/%d/%Y %H:%M:%S")
    return now.strftime("%H:%M:%S")

def get_datasets_models_and_window_sizes():
    """ Get all datasets, models and window sizes available in the static folder. """
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


@app.route("/run_our_drift_experiment")
def run_our_drift_experiment():
    """ Run controlled drift experiment on pre-uploaded use cases. """
    title = 'DriftLens'
    available_data = get_datasets_models_and_window_sizes()
    return render_template('run_our_drift_experiment.html', title=title, data=available_data)

def load_embedding(filepath, E_name=None, Y_original_name=None, Y_predicted_name=None, load_original_labels=True):
    """ Load embedding from HDF5 file. """
    if filepath is not None:
        with h5py.File(filepath, "r") as hf:
            if E_name is None:
                E = hf["E"][()]
            else:
                E = hf[E_name][()]
            if load_original_labels:
                if Y_original_name is None:
                    Y_original = hf["Y_original"][()]
                else:
                    Y_original = hf[Y_original_name][()]
            else:
                Y_original = None
            if Y_original_name is None:
                Y_original = hf["Y_original"][()]
            else:
                Y_original = hf[Y_original_name][()]
            if Y_predicted_name is None:
                Y_predicted = hf["Y_predicted"][()]
            else:
                Y_predicted = hf[Y_predicted_name][()]
    else:
        raise Exception("Error in loading the embedding file.")
    return E, Y_original, Y_predicted


def run_drift_detection_background_new_experiment_thread(form_parameters):
    """ Run drift detection in background thread. """
    print("done")
    new_unseen_embedding_path = f"static/new_use_cases/tmp/datastream.hdf5"
    E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(new_unseen_embedding_path, load_original_labels=False)
    print(E_new_unseen.shape)
    dl = DriftLens()
    baseline = dl.load_baseline(folderpath="static/new_use_cases/tmp/",
                     baseline_name="baseline")

    print(Y_predicted_new_unseen)

    selected_batch_threshold = float(form_parameters["hidden_batch_threshold"])
    training_labels_id_list = baseline.label_list
    #training_labels_id_list = [0,1,2]
    n_samples = len(Y_original_new_unseen)
    window_size = int(form_parameters["hidden_window_size"])
    n_windows = n_samples//window_size
    E_windows = []
    Y_predicted_windows = []
    selected_latency = 0

    for i in range(n_windows):
        start_index = i*window_size
        end_index = i * window_size + window_size
        E_windows.append(E_new_unseen[start_index:end_index])
        Y_predicted_windows.append(Y_predicted_new_unseen[start_index:end_index])

    for i, (E_w, y_pred) in enumerate(zip(E_windows, Y_predicted_windows)):
        window_distance = dl.compute_window_distribution_distances(E_w, y_pred)
        window_distance["window_id"] = i
        if isinstance(window_distance["batch"], complex):
            window_distance["batch"] = float(_utils.clear_complex_number(window_distance["batch"]).real)

        if window_distance["batch"] > selected_batch_threshold:
            batch_drift_prediction = 1
        else:
            batch_drift_prediction = 0

        for l in training_labels_id_list:
            if isinstance(window_distance["per-label"][str(l)], complex):
                print("clearing:", window_distance["per-label"][str(l)])
                window_distance["per-label"][str(l)] = float(_utils.clear_complex_number(window_distance["per-label"][str(l)]).real)
                print(window_distance["per-label"][str(l)])
        print(f"window: {i} - {window_distance}")

        per_label_distances = ",".join(str(v) for k,v in window_distance["per-label"].items())
        print(window_distance["per-label"])
        socketio.emit('updateDriftData', {'batch_distance': window_distance["batch"], "per_label_distances":per_label_distances ,
                                           "date": get_current_datetime(), "batch_drift_prediction":batch_drift_prediction, "window_id":i})
        socketio.sleep(selected_latency/1000)

def run_drift_detection_background_thread(form_parameters, config_dict):
    selected_dataset = form_parameters['dataset']
    selected_model = form_parameters['model']
    selected_window_size = int(form_parameters['window_size'])
    selected_drift_pattern = form_parameters['drift_pattern']
    selected_batch_threshold = float(form_parameters['hidden_batch_threshold'])
    print("Selected batch Threshold", selected_batch_threshold)
    # Load Embedding
    new_unseen_embedding_path = f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}/saved_embeddings/new_unseen_embedding.hdf5"
    drifted_embedding_path = f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}/saved_embeddings/drifted_embedding.hdf5"

    E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(new_unseen_embedding_path)
    E_drift, Y_original_drift, Y_predicted_drift = load_embedding(drifted_embedding_path)

    training_labels_id_list = config_dict["training_labels_id_list"]
    training_labels_names_list = config_dict["training_labels_name_list"]
    drift_labels_id_list = config_dict["drift_labels_id_list"]
    drift_labels_names_list = config_dict["drift_labels_name_list"]


    wg = WindowsGenerator(training_labels_id_list, drift_labels_id_list, E_new_unseen, Y_predicted_new_unseen,
                          Y_original_new_unseen, E_drift, Y_predicted_drift, Y_original_drift)

    # Create DriftLens Object
    dl = DriftLens(training_labels_id_list)

    print("Loading Baseline")
    dl.load_baseline(folderpath=f"static/use_cases/datasets/{selected_dataset}/models/{selected_model}", baseline_name="baseline")

    flag_shuffle = True
    flag_replacement = True


    if selected_drift_pattern == "no_drift":
        print("no drift")
        selected_number_of_windows = int(form_parameters["number_of_windows_no_drift"])
        selected_latency = int(form_parameters["latency_no_drift"])
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
        selected_latency = int(form_parameters["latency_sudden_drift"])
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
        selected_latency = int(form_parameters["latency_incremental_drift"])
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
        selected_number_of_windows = int(form_parameters["number_of_windows_periodic_drift"])
        selected_latency = int(form_parameters["latency_periodic_drift"])
        selected_drift_offset = int(form_parameters["drift_offset_periodic_drift"])
        selected_drift_duration = int(form_parameters["drift_duration_periodic_drift"])
        selected_drift_percentage = int(form_parameters["drift_percentage_periodic_drift"]) / 100


        E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_periodic_drift_windows_generation(
            window_size=selected_window_size,
            n_windows=selected_number_of_windows,
            drift_offset=selected_drift_offset,
            drift_duration=selected_drift_duration,
            drift_percentage=selected_drift_percentage,
            flag_shuffle=flag_shuffle,
            flag_replacement=flag_replacement,
            socketio=socketio)


    for i, (E_w, y_pred, y_true) in enumerate(zip(E_windows, Y_predicted_windows, Y_original_windows)):
        window_distance = dl.compute_window_distribution_distances(E_w, y_pred)
        window_distance["window_id"] = i
        if isinstance(window_distance["batch"], complex):
            window_distance["batch"] = float(_utils.clear_complex_number(window_distance["batch"]).real)

        if window_distance["batch"] > selected_batch_threshold:
            batch_drift_prediction = 1
        else:
            batch_drift_prediction = 0

        for l in training_labels_id_list:
            if isinstance(window_distance["per-label"][str(l)], complex):
                print("clearing:", window_distance["per-label"][str(l)])
                window_distance["per-label"][str(l)] = float(_utils.clear_complex_number(window_distance["per-label"][str(l)]).real)
                print(window_distance["per-label"][str(l)])
        print(f"window: {i} - {window_distance}")

        per_label_distances = ",".join(str(v) for k,v in window_distance["per-label"].items())
        print(window_distance["per-label"])
        socketio.emit('updateDriftData', {'batch_distance': window_distance["batch"], "per_label_distances":per_label_distances ,
                                           "date": get_current_datetime(), "batch_drift_prediction":batch_drift_prediction, "window_id":i})
        socketio.sleep(selected_latency/1000)

@app.route("/drift_lens_monitor", methods=["GET", "POST"])
def drift_lens_monitor():
    """ Route to the drift lens monitor page. """
    title = 'DriftLens'

    if request.method == "POST":
        all_parameters = request.form.to_dict()
    global thread
    print('Client connected')

    config_file_path = f'static/use_cases/datasets/{all_parameters["dataset"]}/config.yml'
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            config_dict = yaml.safe_load(f)

    training_labels_names_list = config_dict["training_labels_name_list"]

    global thread
    with thread_lock:
        if thread is None:
            #thread = socketio.start_background_task(background_thread)
            thread = socketio.start_background_task(run_drift_detection_background_thread, all_parameters, config_dict)

    return render_template('drift_lens_monitor.html', title=title, num_labels=len(training_labels_names_list), label_names=",".join(training_labels_names_list))


@app.route("/drift_lens_monitor_new_experiment", methods=["GET", "POST"])
def drift_lens_monitor_new_experiment():
    """ Route to the drift lens monitor page for users uploaded data. """
    title = 'DriftLens'

    if request.method == "POST":
        all_parameters = request.form.to_dict()
        print(all_parameters)
    global thread
    print('Client connected')

    #config_file_path = f'static/use_cases/datasets/{all_parameters["dataset"]}/config.yml'
    #if os.path.exists(config_file_path):
    #    with open(config_file_path) as f:
    #        config_dict = yaml.safe_load(f)

    #training_labels_names_list = config_dict["training_labels_name_list"]
    training_labels_names_list = ["Technology", "Sale-Ads", "sport", "Politics", "Science"]


    global thread
    with thread_lock:
        if thread is None:
            #thread = socketio.start_background_task(background_thread)
            thread = socketio.start_background_task(run_drift_detection_background_new_experiment_thread, all_parameters)

    return render_template('drift_lens_monitor.html', title=title, num_labels=5, label_names=",".join(training_labels_names_list))





"""
Serve root index file
"""
@app.route('/')
def index():
    """ Route to the index page. """
    return render_template('index.html')

@app.route('/documentation')
def documentation():
    """ Route to the documentation page."""
    return render_template('documentation.html')

@app.route('/run_your_drift_experiment',  methods=["GET", "POST"])
def run_your_drift_experiment():
    timestamp = int(time.time())
    session_dir = os.path.join('static', 'new_use_cases', 'tmp', str(timestamp))
    data_dir = os.path.join(session_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return render_template('run_your_drift_experiment.html', data_dir=data_dir)


@app.route('/get_threshold_values', methods=["GET", "POST"])
def get_threshold_values():
    """ Route to get the threshold values for the drift lens monitor. """
    dataset = request.args.get('dataset')
    model = request.args.get('model')
    window_size = request.args.get('window_size', type=int)

    # Print the received parameters for debugging
    print("Received Parameters:")
    print("Dataset:", dataset)
    print("Model:", model)
    print("Window Size:", window_size)

    # Convert data to a list for JSON response
    data_list = np.load(f"static/use_cases/datasets/{dataset}/models/{model}/window_sizes/{window_size}/thresholds/th_batch.npy").tolist()
    return jsonify(data_list)


@app.route('/compute_baseline', methods=['POST'])
def compute_baseline():
    """ Route to compute the baseline. """
    print("--- Computing Baseline")
    baseline_embedding_path = f"static/new_use_cases/tmp/baseline.hdf5"
    batch_n_pc = 150
    per_label_n_pc = 75
    E_baseline, Y_original_baseline, Y_predicted_baseline = load_embedding(baseline_embedding_path, load_original_labels=True)
    label_list = sorted(list(set(Y_predicted_baseline)))
    label_list = [int(l) for l in label_list]
    baseline_estimator = _baseline.StandardBaselineEstimator(label_list, batch_n_pc, per_label_n_pc)
    baseline = baseline_estimator.estimate_baseline(E_baseline, Y_predicted_baseline)
    baseline_path = f"static/new_use_cases/tmp"
    baseline_path = baseline.save(baseline_path, "baseline")
    return jsonify(message="Baseline Estimated")

@app.route('/estimate_threshold', methods=['POST'])
def estimate_threshold():
    """ Route to estimate the threshold. """

    print("--- Estimating Threshold")
    threshold_embedding_path = f"static/new_use_cases/tmp/threshold.hdf5"
    batch_n_pc = 150
    per_label_n_pc = 75
    #training_label_list = [0, 1, 2, 3, 4]
    E_th, Y_original_th, Y_predicted_th = load_embedding(threshold_embedding_path, load_original_labels=True)

    training_label_list = range(max(Y_predicted_th))

    base_path = f"static/new_use_cases/tmp"
    dl = DriftLens(training_label_list)
    dl.load_baseline(base_path, "baseline")

    wg = WindowsGenerator(training_label_list,
                          [max(Y_predicted_th)+1],
                          E_th,
                          Y_predicted_th,
                          Y_original_th,
                          E_th,
                          Y_predicted_th,
                          Y_original_th)

    per_batch_distances = []
    per_label_distances = {label: [] for label in training_label_list}

    for i in range(100):
        E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_without_drift_windows_generation(
            window_size=500,
            n_windows=1,
            flag_shuffle=True,
            flag_replacement=True,
            update_progressbar=False
        )

        distribution_distances = dl.compute_window_list_distribution_distances(E_windows, Y_predicted_windows)

        per_batch_distances.append(distribution_distances[0][0]["batch"])
        for l in training_label_list:
            per_label_distances[l].append(distribution_distances[0][0]["per-label"][str(l)])

    per_batch_distances_arr = np.array(per_batch_distances)
    indices = (-per_batch_distances_arr).argsort()
    per_batch_distances_sorted = per_batch_distances_arr[indices]
    per_batch_distances_sorted = per_batch_distances_sorted + 1.2

    for l in training_label_list:
        per_label_distances[l] = sorted(per_label_distances[l], reverse=True)
        #per_label_distances[l] = per_label_distances[l] + 1.2

    th_path = os.path.join(base_path, "thresholds")

    if not os.path.exists(th_path):
        os.makedirs(th_path)
    else:
        shutil.rmtree(th_path)  # Removes all the subdirectories!
        os.makedirs(th_path)


    with open(os.path.join(th_path, "th_batch.npy"), 'wb') as f:
        np.save(f, per_batch_distances_sorted)

    for l in training_label_list:
        with open(os.path.join(th_path, f'th_label_{l}.npy'), 'wb') as f:
            np.save(f, per_label_distances[l])

    data_list = per_batch_distances_sorted.tolist()
    return jsonify(data_list)

@app.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    """ Route to upload a chunk of a file. """
    chunk = request.files['fileChunk']
    upload_type = request.form['uploadType']
    dataset = request.form['datasetName']
    model = request.form['modelName']
    print(request.form['datasetName'])
    filename = secure_filename(chunk.filename)
    data_dir = "static/new_use_cases/tmp"
    # Define the full path for the file within the session directory
    file_path = os.path.join(data_dir, f'{upload_type}.hdf5')

    # Append the chunk to the file
    with open(file_path, 'ab') as f:
        f.write(chunk.read())

    print("Uploaded chunk for", upload_type)

    # Check if all chunks for the baseline are received
    if upload_type == 'baseline':
        # Combine all chunks into a single baseline file
        combine_chunks(data_dir, 'baseline.hdf5')
    elif upload_type == 'threshold':
        # Combine all chunks into a single threshold file
        combine_chunks(data_dir, 'threshold.hdf5')
    elif upload_type == 'datastream':
        # Combine all chunks into a single datastream file
        combine_chunks(data_dir, 'datastream.hdf5')
    return jsonify(message="Chunk received")


def combine_chunks(data_dir, output_filename):
    """ Combine all chunks into a single file. """
    # Get a list of all chunk files
    chunk_files = [f for f in os.listdir(data_dir) if f.startswith('baseline_chunk_')]
    chunk_files.sort()  # Sort to ensure correct order

    # Combine all chunks into a single file
    with open(os.path.join(data_dir, output_filename), 'ab') as output_file:
        for chunk_file in chunk_files:
            with open(os.path.join(data_dir, chunk_file), 'rb') as chunk:
                output_file.write(chunk.read())

    # Clean up the individual chunk files
    for chunk_file in chunk_files:
        os.remove(os.path.join(data_dir, chunk_file))

    print("Combined baseline chunks into", output_filename)



@socketio.on('disconnect')
def disconnect():
    """ Disconnect the client. """
    print('Client disconnected',  request.sid)

if __name__ == '__main__':
    socketio.run(app)