import torch
from torch.utils.data import Dataset
import datasets


class CoNLL2003Dataset(Dataset):
    def __init__(self, data, tokenizer, tag_to_id):
        self.data = data
        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx]['tokens']
        labels = self.data[idx]['ner_tags']

        tokenized_input = self.tokenizer(words, is_split_into_words=True, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = tokenized_input['input_ids'].squeeze()  # Remove the batch dimension

        # Align the labels with tokenized input
        new_labels = []
        word_ids = tokenized_input.word_ids(batch_index=0)  # Get the word id for each token
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:  # Special tokens or same word as before
                new_labels.append(-100)  # -100 will be ignored in the loss computation
            else:
                new_labels.append(self.tag_to_id[labels[word_idx]])
            previous_word_idx = word_idx

        # Convert new_labels to tensor
        labels = torch.tensor(new_labels, dtype=torch.long)

        return input_ids, labels