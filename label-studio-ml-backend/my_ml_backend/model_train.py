import torch
import os
from _model import LSTMNerModel
from dataset import NERDataset
from torch.utils.data import DataLoader
from utils import process_annotations
import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from config import (
    LABELS,  
    model_version,
)
from utils import load_model_config


def load_dataset(init_model):
    if init_model:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        initial_data = os.path.join(data_dir, "inital_data.json")
        with open(initial_data, "r") as file:
            initial_data = json.load(file)
        processed_annotations = process_annotations(initial_data)
        return processed_annotations
    else:
        init_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        init_data = os.path.join(init_data_dir, "inital_data.json")
        with open(init_data, "r") as file:
            init_data = json.load(file)
        processed_annotations = process_annotations(init_data)

        new_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        new_data = os.path.join(new_data_dir, "new_data.json")
        with open(new_data, "r") as file:
            new_data = json.load(file)
        processed_annotations += process_annotations(new_data)

        print(f"Processed annotations: {processed_annotations}")
        return processed_annotations

def save_model(dir_path, model, word_to_ix, tag_to_ix):
    if (dir_path == "current_model") or (dir_path == "init_model"):
        model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), dir_path
        )
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_save_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), model_save_path)

        mapping_save_path = os.path.join(model_dir, "mappings.pth")
        torch.save(
            {"word_to_ix": word_to_ix, "tag_to_ix": tag_to_ix},
            mapping_save_path,
        )

        print(f"Model and mappings saved to {model_dir}")
    else:
        print("Invalid directory path")
    
def main(init_model, embedding_dim, hidden_dim, bidirectional, dropout_rate, learning_rate, epochs, batch_size, optimizer):
    processed_annotations = load_dataset(init_model)


    word_to_ix = {"<UNK>": 0}
    tag_to_ix = {"O": 0}  # 'O' for outside any named entity
    def pad_collate(batch):
        (xx, yy) = zip(*batch)

        # Convert list of sequences into a padded tensor for both inputs and targets
        xx_pad = pad_sequence(
            [torch.tensor(x) for x in xx],
            batch_first=True,
            padding_value=word_to_ix["<UNK>"],
        )
        yy_pad = pad_sequence(
            [torch.tensor(y) for y in yy], batch_first=True, padding_value=tag_to_ix["O"]
        )

        return xx_pad, yy_pad

    # Assuming LABELS is a list of all unique labels in your dataset
    for label in LABELS:
        if f"B-{label}" not in tag_to_ix:
            tag_to_ix[f"B-{label}"] = len(tag_to_ix)
        if f"I-{label}" not in tag_to_ix:
            tag_to_ix[f"I-{label}"] = len(tag_to_ix)

    for sentence, tags in processed_annotations:
        #print(f"Sentence: {sentence}")
        #print(f"Tags: {tags}")
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    # Initialize the dataset and dataloader
    dataset = NERDataset(processed_annotations, word_to_ix, tag_to_ix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate
    )


    # Initialize the model
    model = LSTMNerModel(
        vocab_size=len(word_to_ix),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        tagset_size=len(tag_to_ix),
        bidirectional=bidirectional,
        dropout_rate=dropout_rate,
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()

    if optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print("Invalid optimizer. Defaulting to Adam optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for sentence, tags in dataloader:
            model.zero_grad()

            sentence_in = sentence.to(device)
            targets = tags.to(device)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        #print(f"Epoch: {epoch}, Loss: {total_loss / len(dataloader)}")

    # Save the model and mappings
    if init_model:
        save_model("init_model", model, word_to_ix, tag_to_ix)
    else:
        save_model("current_model", model, word_to_ix, tag_to_ix)
    
    return model, word_to_ix, tag_to_ix
