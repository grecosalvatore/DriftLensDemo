from pathlib import Path
import json

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def clear_complex_numbers(scores):
    return [complex(score).real if type(score) == str else score for score in scores]

def clear_complex_number(score):
    return complex(score).real if type(score) == str else score

def increase_data_generation_progress_bar(socketio, percentage):
    # Create a message with the current percentage
    message = json.dumps({"currentProgress": percentage})
    socketio.emit('UpdateProgressBarDataStreamGeneration', message)
    return