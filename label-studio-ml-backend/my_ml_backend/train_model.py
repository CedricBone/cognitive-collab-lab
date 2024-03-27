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
import torch
import torch.optim as optim
from transformers import BertTokenizerFast


# Function to process a single annotation instance for fine-tuning
def process_annotation(annotation, tokenizer, tag_to_id):
    print(f"Processing annotation: {annotation}")
    sentence, annotations = annotation
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Initialize label_ids with -100 to ignore non-first subword tokens in labels
    label_ids = [-100] * len(tokens)

    # Align tokens with labels
    current_position = 0
    for start, end, _, label in annotations:
        label = "B-" + label
        label_id = tag_to_id[label]
        token_start = len(tokenizer.tokenize(sentence[:start]))
        token_end = len(tokenizer.tokenize(sentence[:end]))

        # Sometimes the tokenizer adds special tokens; adjust indices accordingly
        adjust = (tokens[0] == tokenizer.cls_token)

        for i in range(token_start + adjust, token_end + adjust):
            if i < len(label_ids):
                label_ids[i] = label_id

    # Convert to tensors
    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    label_ids_tensor = torch.tensor([label_ids], dtype=torch.long)

    return token_ids_tensor, label_ids_tensor


def fine_tune_model2(model, annotation, tag_to_id, optimizer_type="adam", learning_rate=0.01, grad_clip=None, weight_decay=0.0, momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8, rho=0.9):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    model.train()  # Set model to training mode

    num_labels = len(tag_to_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_sentence_tensor, label_tensor = process_annotation(annotation[0], tokenizer, tag_to_id)
    tokenized_sentence_tensor, label_tensor = tokenized_sentence_tensor.to(device), label_tensor.to(device)

    # Initialize the optimizer with additional parameters
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=rho, eps=eps, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=rho, eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    else:
        print(f"Optimizer type '{optimizer_type}' is not recognized. Defaulting to Adam optimizer.")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

    # Forward pass
    tag_scores = model(tokenized_sentence_tensor)

    # Calculate loss
    loss = torch.nn.functional.nll_loss(tag_scores.view(-1, num_labels), label_tensor.view(-1), ignore_index=-100)

    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()

    # Apply gradient clipping if specified
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    print(loss.item())

    return model


