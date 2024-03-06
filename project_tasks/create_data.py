import json
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset
import os


def extract_test_sentences_to_json(dataset_name, output_files):
    # Load the dataset's test split
    dataset = load_dataset(dataset_name, split='test')
    
    # Extract sentences and format them
    formatted_sentences = [{"text": " ".join(sentence['tokens'])} for sentence in dataset]
    length = len(formatted_sentences)
    part_len = length // 4 

    # Save the formatted data to a JSON file
    with open(output_files[0], 'w') as f:
        json.dump(formatted_sentences[0:part_len], f, ensure_ascii=False, indent=2)
        print(f"Data saved to {output_files[0]}")
    with open(output_files[1], 'w') as f:
        json.dump(formatted_sentences[part_len:part_len*2], f, ensure_ascii=False, indent=2)
        print(f"Data saved to {output_files[1]}")
    with open(output_files[2], 'w') as f:
        json.dump(formatted_sentences[part_len*2:part_len*3], f, ensure_ascii=False, indent=2)
        print(f"Data saved to {output_files[2]}")
    with open(output_files[3], 'w') as f:
        json.dump(formatted_sentences[part_len*3:], f, ensure_ascii=False, indent=2)
        print(f"Data saved to {output_files[3]}")
    with open(output_files[4], 'w') as f:
        json.dump(formatted_sentences, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {output_files[4]}")

    
extract_test_sentences_to_json("conll2003", ["solo.json", "solo_con.json", "pair.json", "pair_con.json", "all_tasks.json"])



