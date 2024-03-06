import os
import json
import requests
import time
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk import Client
from torch.cuda.amp import GradScaler, autocast
from transformers import BertTokenizerFast
from _model import SimpleNERModel
from dataset import CoNLL2003Dataset

import train_model2
import predict_model2
from utils import (
    print_project_ids,
    load_model_config,
    process_annotations,
    list_predictions,
    delete_prediction,
    create_prediction,
    get_all_annotations,
    initialize_new_data_file,
    convert_predictions_to_label_studio_format,
)
from config import (
    LABEL_STUDIO_URL,
    LABEL_STUDIO_API_KEY,
    base_url,
    headers,
    projectID,
    LABELS,
    model_version,
)


ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
ls.check_connection()
print_project_ids(ls)
project = ls.get_project(projectID)


class NewModel(LabelStudioMLBase):
    def __init__(self, project_id=projectID, **kwargs):
        super().__init__(**kwargs)
        #load model params
        config = load_model_config(init=True)
        self.embedding_dim = int(config["embedding_dim"])
        self.hidden_dim = int(config["hidden_dim"])
        self.bidirectional = True
        self.dropout_rate = float(config["dropout_rate"])
        self.learning_rate = float(config["learning_rate"])
        self.epochs = int(config["epochs"])
        self.batch_size = int(config["batch_size"])
        self.optimizer = config["optimizer"]
        self.grad_clip = config["grad_clip"]
        self.weight_decay = config["weight_decay"] 
        self.momentum = config["momentum"]
        self.beta1 = config["beta1"]
        self.beta2 = config["beta2"] 
        self.eps = config["eps"]
        self.rho = config["rho"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load mappings
        self.tag_to_id = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_model", "tag_to_id.pth"))
        self.id_to_tag = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_model", "id_to_tag.pth"))
        # Initialize tokenizer and calculate vocab size and number of labels
        self.tokenizer = train_model2.BertTokenizerFast.from_pretrained('bert-base-cased')
        self.num_labels = len(self.tag_to_id)
        self.vocab_size = self.tokenizer.vocab_size
        # Initialize the model
        model = train_model2.SimpleNERModel(self.vocab_size, self.embedding_dim, self.hidden_dim, self.num_labels).to(self.device)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_model", "model.pth")
        model.load_state_dict(torch.load(model_path))
        self.model = model
        print(f"Current model loaded from {model_path}")

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            task_id = task["id"]
            original_text = task["data"]["text"]
            sentences = [original_text]  # Assuming each task contains a single sentence for simplicity


            # Get predictions for each sentence
            sentence_predictions = predict_model2.predict_entities_adjusted(sentences, self.model, self.id_to_tag)
            print(f"Predictions pre removal: {sentence_predictions}")
            # replace 'B-Miscellaneous' or 'I-Miscellaneous' with 'O'
            for i in range(len(sentence_predictions)):
                for j in range(len(sentence_predictions[i])):
                    if sentence_predictions[i][j][1] == 'B-Miscellaneous' or sentence_predictions[i][j][1] == 'I-Miscellaneous':
                        sentence_predictions[i][j] = (sentence_predictions[i][j][0], 'O')
            print(f"Predictions post removal: {sentence_predictions}")
            
            # Convert predictions to Label Studio format
            ls_results = convert_predictions_to_label_studio_format(sentence_predictions[0], original_text)
            #print(f"Predictions: {ls_results}")


            predictions.append({
                "result": ls_results,
                "task": task_id
            })


        return predictions



    def fine_tune(self, annotation, **kwargs):
        config = load_model_config(init=False)
        self.dropout_rate = float(config["dropout_rate"])
        self.learning_rate = float(config["learning_rate"])
        self.epochs = int(config["epochs"])
        self.batch_size = int(config["batch_size"])
        self.optimizer = config["optimizer"].lower()
        grad_clip = config["grad_clip"]
        if grad_clip == "None":
            self.grad_clip = None
        else:
            self.grad_clip = float(config["grad_clip"])
        self.weight_decay = float(config["weight_decay"])
        self.momentum = float(config["momentum"])
        self.beta1 = float(config["beta1"])
        self.beta2 = float(config["beta2"])
        self.eps = float(config["eps"])
        self.rho = float(config["rho"])
        self.model = train_model2.fine_tune_model2(self.model, annotation, self.tag_to_id, optimizer_type=self.optimizer, learning_rate=self.learning_rate, grad_clip=self.grad_clip, weight_decay=self.weight_decay, momentum=self.momentum, beta1=self.beta1, beta2=self.beta2, eps=self.eps, rho=self.rho)

    def fit(self, event, data, **kwargs):
        print(f"Event: {event}")

        if (event == "ANNOTATION_UPDATED") or (event == "ANNOTATION_CREATED"):
            processed_annotations = process_annotations([data])  # Assuming data is a single annotation
            print(f"{event}: {processed_annotations}")
            self.fine_tune(processed_annotations)
            self.update_predictions()
        else:
            print("Event not related to annotation creation or update. No action taken.")

        return {
            "model_path": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "current_model", "model.pth"
            )
        }

    def update_predictions(self):
        tasks = project.get_unlabeled_tasks()
        tasks = tasks[:2]
        for task in tasks:
            task_id = task["id"]
            # Assuming 'get_task_predictions' is a method that retrieves predictions for a specific task
            existing_predictions = task.get("predictions", [])

            # Generate prediction result for the current task
            prediction_result = self.predict([task])[0]

            if existing_predictions:
                # Assuming there's only one prediction per task for simplicity
                existing_prediction_id = existing_predictions[0]["id"]
                update_url = (
                    f"{LABEL_STUDIO_URL}/api/predictions/{existing_prediction_id}/"
                )
                data = {
                    "model_version": model_version,
                    "result": prediction_result["result"],
                    "score": prediction_result.get("score", 1.0),
                    "cluster": prediction_result.get("cluster", None),
                    "task": task_id,
                }
                response = requests.patch(update_url, json=data, headers=headers)
                if response.status_code not in [200, 204]:
                    print(
                        f"Failed to update prediction for task {task_id}. Status code: {response.status_code}"
                    )
            else:
                # If there's no existing prediction, create a new one
                # Assuming 'create_prediction' is a method that correctly formats and sends a request to create a new prediction
                if not create_prediction(task_id, prediction_result, projectID):
                    print(f"Failed to create prediction for task {task_id}.")

        print("Predictions updated.\n\n\n")
