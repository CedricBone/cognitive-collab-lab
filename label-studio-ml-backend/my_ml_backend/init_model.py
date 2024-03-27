import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from transformers import BertTokenizerFast
import os
from dataset import CoNLL2003Dataset
from _model import SimpleNERModel
from utils import load_model_config
import shutil
import time
   
def collate_fn(batch):
    input_ids, labels = zip(*batch)  # Unzip the batch to separate sequences and labels

    # Pad the sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for labels padding

    return input_ids_padded, labels_padded

def train(model, data_loader, optimizer, device, num_labels):
    model.train()
    total_loss = 0
    for sentence, labels in data_loader:
        sentence, labels = sentence.to(device), labels.to(device)
        model.zero_grad()
        tag_scores = model(sentence)
        loss = nn.functional.nll_loss(tag_scores.view(-1, num_labels), labels.view(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def train_init_model(embedding_dim, hidden_dim, optimizer, batch_size, num_epochs, learning_rate=0.01):
    # Load the dataset
    dataset = load_dataset("conll2003")

    # Tag to ID mapping
    unique_tags = set(tag for doc in dataset["train"]["ner_tags"] for tag in doc)
    tag_to_id = {tag: id for id, tag in enumerate(unique_tags)}

    # Assuming `dataset` is your loaded dataset
    sentences = [" ".join(sentence['tokens']) for sentence in dataset['train']]
    # Replace BasicTokenizer with BertTokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    # Parameters

    num_labels = len(tag_to_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    train_data = CoNLL2003Dataset(dataset['train'], tokenizer, tag_to_id)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model, Loss, and
    vocab_size = tokenizer.vocab_size
    print(f"vocab size: {vocab_size}")
    model = SimpleNERModel(vocab_size, embedding_dim, hidden_dim, num_labels).to(device)

    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print("Invalid optimizer. Defaulting to Adam optimizer")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Person, Location, Organization, Miscillaneous
    tag_to_id = {'O': 0, 'B-Person': 1, 'I-Person': 2, 'B-Location': 3, 'I-Location': 4, 'B-Organization': 5, 'I-Organization': 6, 'B-Miscellaneous': 7, 'I-Miscellaneous': 8}
    id_to_tag = {id: tag for tag, id in tag_to_id.items()}

    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, device, num_labels)
        print(f"Epoch {epoch+1}, Loss: {loss}")

    # Clear "current_model" directory
    os
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_model")
    print(f"Save path: {save_path}")
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    print("Model saved")
    torch.save(tag_to_id, os.path.join(save_path, "tag_to_id.pth"))
    print("Tag to ID saved")
    torch.save(id_to_tag, os.path.join(save_path, "id_to_tag.pth"))
    print("ID to Tag saved")

    return model, tag_to_id, id_to_tag

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def move_contents(src_directory, dest_directory):
    for filename in os.listdir(src_directory):
        src_path = os.path.join(src_directory, filename)
        dest_path = os.path.join(dest_directory, filename)
        try:
            if os.path.isdir(src_path):
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)
        except Exception as e:
            print('Failed to move %s to %s. Reason: %s' % (src_path, dest_directory, e))

def retrain_model():
    config = load_model_config(init=True)
    embedding_dim = int(config["embedding_dim"])
    hidden_dim = int(config["hidden_dim"])
    bidirectional = True
    dropout_rate = float(config["dropout_rate"])
    learning_rate = float(config["learning_rate"])
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    optimizer = config["optimizer"]
    print(f"Config: {config}")
    model, tag_to_id, id_to_tag = train_init_model(embedding_dim, hidden_dim, optimizer, batch_size, epochs, learning_rate)

def reset_model():
    current_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_model")
    init_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_model")
    clear_directory(current_model_path)
    move_contents(init_model_path, current_model_path)
    print("Model directory updated from init_model to current_model.")

# Move model to a new folder in the models directory (E:\cognitive_collab_iML2\models)
def move_model(trial_type, participant_id):
    current_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_model")
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    new_model_path = os.path.join(models_path, f"{trial_type}_{participant_id}_model_{time.strftime('%Y%m%d-%H%M%S')}")
    if os.path.exists(new_model_path):
        shutil.rmtree(new_model_path)
    shutil.copytree(current_model_path, new_model_path)
    print(f"Model moved to {new_model_path}")

if __name__ == "__main__":

    retrain = input("Do you want to retrain the model? (y/n): ")
    if retrain.lower().strip() == "y":
        retrain_model()
    else:
        reset_model()
