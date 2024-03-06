import os
import json
import requests
from config import (
    LABEL_STUDIO_URL,
    LABEL_STUDIO_API_KEY,
    base_url,
    headers,
    projectID,
    LABELS,
    model_version,
)

def convert_predictions_to_label_studio_format(predictions, text):
    ls_results = []
    start_index = 0
    for word, tag in predictions:
        end_index = start_index + len(word)
        if tag != 'O':  # Ignore 'O' tags
            label = tag.split('-')[-1]  # Get the actual label without the I- or B- prefix
            ls_results.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start_index,
                    "end": end_index,
                    "text": word,
                    "labels": [label]
                }
            })
        start_index = text.find(word, start_index) + len(word) + 1  # Update start_index for next iteration
    return ls_results

def print_project_ids(ls):
    print("Project IDs:")
    for project in ls.get_projects():
        print(project.id)


def load_model_config(init=False):
    if init:  
        config_path = os.path.join(os.path.dirname(__file__), "init_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config = json.load(file)
            return config
        else:
            return {}
    else:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config = json.load(file)
            return config
        else:
            return {}


def list_predictions():
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to list predictions. Status code:", response.status_code)
        return None


# Function to delete a prediction
def delete_prediction(prediction_id):
    delete_url = f"{base_url}{prediction_id}/"
    response = requests.delete(delete_url, headers=headers)
    if response.status_code != 204:
        print(
            f"Failed to delete prediction with ID {prediction_id}. Status code:",
            response.status_code,
        )


def create_prediction(task_id, prediction_data, project_id):
    create_url = f"{LABEL_STUDIO_URL}/api/predictions/"
    data = {
        "task": task_id,
        "result": prediction_data["result"],
        "score": prediction_data.get("score", 1.0),
        "model_version": f"{model_version}",
    }
    response = requests.post(create_url, json=data, headers=headers)
    if response.status_code in [200, 201]:
        print(f"Prediction created for task {task_id}.")
    else:
        print(
            f"Failed to create prediction for task {task_id}. Status code: {response.status_code}"
        )


# "http://api.labelstud.io/api/tasks/{id}/annotations/"
def get_annotation(task_id):
    url = f"{LABEL_STUDIO_URL}/api/tasks/{task_id}/annotations/"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Failed to get annotations for task {task_id}. Status code: {response.status_code}"
        )
        return None


def get_all_annotations(project):
    tasks = [task for task in project.get_tasks_ids()]
    annotations = []
    for task in tasks:
        annotations.append(get_annotation(task))
    return annotations


def process_annotations(data):
    processed_data = []

    # Check if data is a dictionary (single annotation case), and wrap it in a list
    if isinstance(data, dict):
        # This assumes the dictionary structure is similar to the ANNOTATION_UPDATED event structure
        if "annotation" in data and "task" in data:
            # Directly use the provided structure to create a single-item list
            data = [data]
        else:
            # If the structure is not recognized, return an empty list to avoid processing errors
            return processed_data

    # If data is already a list, directly proceed to process each item
    for item in data:
        text = ""
        entities = []
        # Handling structure from loaded JSON files (initial and new data)
        if "data" in item and "annotations" in item:
            text = item["data"]["text"]
            annotations = item["annotations"]
        # Handling structure from an annotation event
        elif "annotation" in item and "task" in item:
            text = item["task"]["data"]["text"]
            annotations = [item["annotation"]]
        else:
            continue  # Skip if neither structure matches

        for annotation in annotations:
            for result in annotation.get("result", []):
                value = result.get("value", {})
                start = value.get("start")
                end = value.get("end")
                entity_text = value.get("text", "")
                labels = value.get("labels", [])
                if labels:
                    label = labels[0]
                    entities.append((start, end, entity_text, label))

        processed_data.append((text, entities))

    return processed_data

def initialize_new_data_file():
    new_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "new_data.json")
    with open(new_data_path, 'w') as file:
        json.dump([], file)  # Initialize the file with an empty list

