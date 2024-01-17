import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk import Client

LABEL_STUDIO_URL = 'http://localhost:8080'
LABEL_STUDIO_API_KEY = '5e185926c1b42768a90baf72ae9d13994dab25b0'
ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
ls.check_connection()
projectID = 2
project = ls.get_project(projectID)

LABELS = ["PER", "ORG", "LOC", "MISC"]

# Dataset class for NER
class NERDataset(Dataset):
   def __init__(self, annotations, word_to_ix, tag_to_ix):
       self.annotations = annotations
       self.word_to_ix = word_to_ix
       self.tag_to_ix = tag_to_ix


   def __len__(self):
       return len(self.annotations)


   def __getitem__(self, idx):
       sentence, entities = self.annotations[idx]
       tokens = sentence.split()
       labels = ["O"] * len(tokens)


       current_char = 0  # To keep track of character position in the sentence
       for token in tokens:
           token_len = len(token)


           for start, end, _, label in entities:
               # Check if the entity starts or ends within the current token
               if current_char <= start < current_char + token_len:
                   labels[tokens.index(token)] = "B-" + label
               elif current_char < end <= current_char + token_len:
                   labels[tokens.index(token)] = "I-" + label


           current_char += token_len + 1  # +1 for the space


       token_indices = [
           self.word_to_ix.get(token, self.word_to_ix["<UNK>"]) for token in tokens
       ]
       label_indices = [self.tag_to_ix[label] for label in labels]
       return torch.tensor(token_indices, dtype=torch.long), torch.tensor(
           label_indices, dtype=torch.long
       )

# Simple LSTM Model for NER
class LSTMNerModel(nn.Module):
   def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
       super(LSTMNerModel, self).__init__()
       self.hidden_dim = hidden_dim
       self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
       self.lstm = nn.LSTM(embedding_dim, hidden_dim)
       self.hidden2tag = nn.Linear(hidden_dim, tagset_size)


   def forward(self, sentence):
       embeds = self.word_embeddings(sentence)
       # Reshape the embedding output to [sequence_length, batch_size, embedding_dim]
       lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], embeds.shape[0], -1))
       tag_space = self.hidden2tag(lstm_out.view(embeds.shape[1], -1))
       tag_scores = torch.log_softmax(tag_space, dim=1)


       return tag_scores

# NER Model class
class NERModel(LabelStudioMLBase):
   def __init__(self, project_id=0, **kwargs):
        super().__init__(**kwargs)
        mappings = torch.load("iml_backend\mappings.pth")
        self.word_to_ix = mappings["word_to_ix"]
        self.tag_to_ix = mappings["tag_to_ix"]

        # Initialize the model with the correct architecture
        self.model = LSTMNerModel(
            vocab_size=len(self.word_to_ix),
            embedding_dim=32,  # These should match the values used during training
            hidden_dim=32,
            tagset_size=len(self.tag_to_ix),
        )

        # Load the state dictionary into the model
        self.model.load_state_dict(torch.load("iml_backend\model.pth"))

   """
   predict() is the function that makes predictions on new data. It takes a
   list of tasks as input and returns a list of results.
   """
   def predict(self, tasks, **kwargs):
       # Extract texts from tasks
       texts = [task["data"]["text"] for task in tasks]
       ids = [task["id"] for task in tasks]


       # Prepare data for prediction
       processed_texts = [
           [
               self.word_to_ix.get(word, self.word_to_ix["<UNK>"])
               for word in text.split()
           ]
           for text in texts
       ]


       # Convert to tensor for model input
       tensor_texts = [
           torch.tensor(text, dtype=torch.long) for text in processed_texts
       ]


       results = []
       for text_tensor, original_text, task_id in zip(tensor_texts, texts, ids):
           # Get predictions from model
           with torch.no_grad():
               tag_scores = self.model(text_tensor.unsqueeze(0))  # Add batch dimension
           predicted_tags = torch.argmax(tag_scores, dim=1)


           # Convert predicted tags to labels
           predicted_labels = [
               list(self.tag_to_ix.keys())[list(self.tag_to_ix.values()).index(tag)]
               for tag in predicted_tags
           ]


           # Convert predictions to Label Studio format
           ls_result = []
           current_label = None
           token_index = 0
           char_index = 0
           tokens = original_text.split()
           for i, label in enumerate(predicted_labels):
               start_char_index = original_text.find(tokens[i], char_index)
               end_char_index = start_char_index + len(tokens[i])


               if label.startswith("B-"):
                   if current_label is not None:
                       ls_result[-1]["value"]["text"] = original_text[
                           ls_result[-1]["value"]["start"] : char_index
                       ].strip()
                   current_label = label[2:]
                   ls_result.append(
                       {
                           "from_name": "label",
                           "to_name": "text",
                           "type": "labels",
                           "value": {
                               "labels": [current_label],
                               "start": start_char_index,
                               "end": end_char_index,
                               "text": "",
                           },
                       }
                   )
               elif label.startswith("I-") and current_label is not None:
                   continue
               else:
                   if current_label is not None:
                       ls_result[-1]["value"]["text"] = original_text[
                           ls_result[-1]["value"]["start"] : char_index
                       ].strip()
                   current_label = None


               char_index = end_char_index + 1  # Update char_index for the next token


           if current_label is not None:
               ls_result[-1]["value"]["text"] = original_text[
                   ls_result[-1]["value"]["start"] : char_index
               ].strip()


           results.append(
               {
                   "result": ls_result,
                   "score": 1.0,
                   "cluster": None,
                   "task": task_id,
                   "from_name": "label",
                   "to_name": "text",
                   "type": "labels",
                   "origin": "prediction",
               }
           )


       return results

   """
   fit() is the function that trains your model. It takes an event and a path
   to the Label Studio data directory as input and returns a dictionary with
   the path to the saved model.
   """
   def fit(self, event, data, **kwargs):
       # Process annotations
       if event == "ANNOTATION_UPDATED":
           annotation_data = data.get("annotation", {})
           task_data = data.get("task", {}).get("data", {})
           text = task_data.get("text", "")
           entities = []
           for result in annotation_data.get("result", []):
               value = result.get("value", {})
               start = value.get("start")
               end = value.get("end")
               entity_text = value.get("text", "")
               labels = value.get("labels", [])
               if labels:
                   label = labels[0]  # assuming each entity has only one label
                   entities.append((start, end, entity_text, label))
           processed_annotations = [(text, entities)]
           self.update_predictions()
       elif event == "ANNOTATION_CREATED":
           annotation_data = data.get("annotation", {})
           task_data = data.get("task", {}).get("data", {})
           text = task_data.get("text", "")
           entities = []
           for result in annotation_data.get("result", []):
               value = result.get("value", {})
               start = value.get("start")
               end = value.get("end")
               entity_text = value.get("text", "")
               labels = value.get("labels", [])
               if labels:
                   label = labels[0]  # assuming each entity has only one label
                   entities.append((start, end, entity_text, label))
           processed_annotations = [(text, entities)]
           self.update_predictions()
       else:
           processed_annotations = self.process_annotations(data)


       # Build word_to_ix and tag_to_ix mappings
       word_to_ix = {"<UNK>": 0}  # Unknown word token
       tag_to_ix = {"O": 0}  # Outside of named entity token


       # Add B- and I- prefixes for each label
       for label in LABELS:
           b_label = "B-" + label
           i_label = "I-" + label
           if b_label not in tag_to_ix:
               tag_to_ix[b_label] = len(tag_to_ix)
           if i_label not in tag_to_ix:
               tag_to_ix[i_label] = len(tag_to_ix)


       # Continue with word_to_ix
       for _, entities in processed_annotations:
           for start, end, text, label in entities:
               words = text.split()
               for word in words:
                   if word not in word_to_ix:
                       word_to_ix[word] = len(word_to_ix)


       # Create dataset
       dataset = NERDataset(processed_annotations, word_to_ix, tag_to_ix)


       # Create dataloader
       dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


       # Initialize model
       model = LSTMNerModel(
           vocab_size=len(word_to_ix),
           embedding_dim=32,
           hidden_dim=32,
           tagset_size=len(tag_to_ix),
       )


       # Train model
       loss_function = nn.NLLLoss()
       optimizer = optim.SGD(model.parameters(), lr=0.1)


       for epoch in range(300):
           total_loss = 0
           for i, (sentence, tags) in enumerate(dataloader):
               model.zero_grad()
               tags = tags.view(-1)  # Flatten the tags tensor
               tag_scores = model(sentence)
               loss = loss_function(tag_scores, tags)
               loss.backward()
               optimizer.step()
               total_loss += loss.item()


       # Save model
       model_save_path = "iml_backend\model.pth"
       torch.save(model.state_dict(), model_save_path)
       mapping_save_path = "iml_backend\mappings.pth"
       torch.save(
           {"word_to_ix": word_to_ix, "tag_to_ix": tag_to_ix}, mapping_save_path
       )


       return {"model_path": model_save_path}

   """
   process_annotations() is a helper function that processes the annotations
   from the Label Studio JSON format into a format that the model can use.
   It takes a path to the Label Studio data directory as input and returns
   a list of processed annotations.
   """
   def process_annotations(self, data):
       with open(data + "train_data.json", "r") as file:
           train_data = json.load(file)

       processed_data = []
       for item in train_data:
           text = item["data"]["text"]
           entities = []
           for annotation in item.get("annotations", []):
               for result in annotation.get("result", []):
                   entity = result.get("value", {})
                   if "start" in entity and "end" in entity and "labels" in entity:
                       start = entity["start"]
                       end = entity["end"]
                       label = entity["labels"][
                           0
                       ]  # assuming each entity has only one label
                       entities.append((start, end, text[start:end], label))

           processed_data.append((text, entities))

       return processed_data
   
   def update_predictions(self):
    tasks = []
    for taskID in project.get_tasks_ids():
        task = project.get_task(taskID)
        tasks.append(task)

    predictions = self.predict(tasks)
    project.create_predictions(predictions)



