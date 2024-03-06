import torch


def main(model, word_to_ix, tag_to_ix, tasks):
    device='cpu'
    ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}  # Inverse mapping for tags

    # Ensure the model is in evaluation mode
    model.eval()

    results = []  # This will hold all the task predictions in Label Studio format
    with torch.no_grad():  # No need to track gradients
        for task in tasks:
            task_id = task["id"]
            original_text = task["data"]["text"]
            words = original_text.split()  # Simple tokenization; adjust as necessary
            
            # Convert words to their index values
            inputs = [word_to_ix.get(word, word_to_ix["<UNK>"]) for word in words]
            inputs_tensor = torch.tensor(inputs, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension and send to device
            
            # Get the model's predictions
            tag_scores = model(inputs_tensor)
            tag_indices = torch.argmax(tag_scores, dim=2)  # Get the most likely tag index for each word
            
            # Convert indices back to tags
            predicted_labels = [ix_to_tag[ix.item()] for ix in tag_indices[0]]
            
            # Convert predictions to Label Studio format
            ls_result = []
            current_label = None
            char_index = 0
            for i, label in enumerate(predicted_labels):
                start_char_index = original_text.find(words[i], char_index)
                end_char_index = start_char_index + len(words[i])

                if label.startswith("B-"):
                    if current_label is not None:
                        ls_result[-1]["value"]["text"] = original_text[
                            ls_result[-1]["value"]["start"]:char_index
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
                            ls_result[-1]["value"]["start"]:char_index
                        ].strip()
                    current_label = None

                char_index = end_char_index + 1  # Update char_index for the next token

            if current_label is not None:
                ls_result[-1]["value"]["text"] = original_text[
                    ls_result[-1]["value"]["start"]:char_index
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

    print(f"Predictions: {results}")
    return results