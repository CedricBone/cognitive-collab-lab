import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from transformers import BertTokenizerFast
from dataset import CoNLL2003Dataset
from _model import SimpleNERModel


def aggregate_predictions(sentence_tokens, predicted_tags):
    aggregated_predictions = []
    current_word = ""
    current_label = ""

    for token, label in zip(sentence_tokens, predicted_tags):
        # Check if the token is a continuation of the previous token
        if token.startswith("##"):
            current_word += token[2:]  # Append without "##"
        else:
            # If there's a current word being built, add it to the predictions
            if current_word:
                aggregated_predictions.append((current_word, current_label))
            current_word = token
            current_label = label

    # Don't forget to add the last word
    if current_word:
        aggregated_predictions.append((current_word, current_label))

    return aggregated_predictions

def predict_entities_adjusted(sentences, model, id_to_tag):
    model.eval()  # Ensure model is in evaluation mode
    predictions = []
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for sentence in sentences:
        # Tokenize the sentence
        tokens = tokenizer.tokenize(sentence)
        # Convert tokens to their IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Convert to tensor and move to the appropriate device
        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)


        with torch.no_grad():
            # Get model predictions
            output = model(token_ids_tensor)
            logits = output if isinstance(output, torch.Tensor) else output[0]
            predicted_tag_ids = torch.argmax(logits, dim=2)
            # Convert predicted tag IDs to labels
            predicted_tags = [id_to_tag[id.item()] for id in predicted_tag_ids[0]]


        # Aggregate predictions to ensure each word has one prediction
        aggregated_predictions = aggregate_predictions(tokens, predicted_tags)
        predictions.append(aggregated_predictions)

    print(predictions)
    return predictions