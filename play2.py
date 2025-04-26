import json, time
import re
from tqdm import tqdm
from openai import OpenAI

def calculate_span_indices(model_output_text, labels):
    """
    Given a model output text and a list of labels, extract and correct span indices if necessary.

    Args:
        model_output_text (str): The text from which to extract spans.
        labels (list): A list of either dicts (soft labels) or lists/tuples (hard labels).

    Returns:
        list: A list of extracted spans with corrected indices.
    """
    span_indices = []

    for label in labels:
        if isinstance(label, dict):  # Soft labels
            text = label["text"]
            start, end = label["start"], label["end"]
            prob = label.get("prob", 1.0)  # Default probability to 1.0 if not provided

            # If indices are -1, find them in the model_output_text
            
            start = model_output_text.find(text)
            if start != -1:
                end = start + len(text)
            else:
                raise ValueError(f"Text '{text}' not found in model_output_text.")

            # Append with recalculated indices
            span_indices.append({"text": text, "start": start, "end": end, "prob": prob})

    return span_indices

def update_hard_labels_from_soft(soft_labels):
    """
    Updates hard labels using soft label indices if they were -1, -1.

    Args:
        soft_labels (list): The processed soft labels containing updated indices.

    Returns:
        list: Updated hard labels as [start, end] lists.
    """
    return [[soft["start"], soft["end"]] for soft in soft_labels]

def generate_labels_with_reasoner(input_data):
    """
    Generates soft and hard labels using the DeepSeek-R1 reasoner model with a progress bar.
    """
    labeled_data = []
    ct = 0
    for entry in tqdm(input_data, desc="Processing entries"):
        try:
            
            # # Add labels to the entry
            # labeled_entry = {
            #     "id": entry["id"],
            #     # "model_id": entry["model_id"],
            #     "lang": entry["lang"],
            #     "model_input": entry['model_input'],
            #     "model_output_text": entry['model_output_text'],
            #     "soft_labels": calculate_span_indices(entry["model_output_text"], entry["soft_labels"]),
            #     "hard_labels": update_hard_labels_from_soft(calculate_span_indices(entry["model_output_text"], entry["soft_labels"]))
            #     # "model_output_logits": entry["model_output_logits"],
            #     # "model_output_tokens": entry["model_output_tokens"]
            # }
            # labeled_data.append(labeled_entry)

            print("Question: ", entry["model_input"])
            print("Answer: ", entry["model_output_text"])
            print("labels:::::")
            for i in entry['hard_labels']:
                print(entry["model_output_text"][i[0]:i[1]])
            print("======================================================================")
            
        except Exception as e:
             print(f"Error processing entry x: {e}")

    return labeled_data


def label_and_save_data(file_path, output_path):
    """
    Load unlabeled data, generate labels, and save the labeled dataset.
    """
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]
    # input_data = input_data[:1]  # Process a subset for testing
    
    print("Generating labels with DeepSeek-R1...")
    labeled_data = generate_labels_with_reasoner(input_data)

    print(f"Saving labeled data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in labeled_data:
            json.dump(entry, f)
            f.write("\n")

    print("Labeling complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate labels using DeepSeek-R1.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the labeled JSONL file.")

    args = parser.parse_args()
    label_and_save_data(args.input, args.output)
