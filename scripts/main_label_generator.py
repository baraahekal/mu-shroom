import json
import logging
import argparse
from tqdm import tqdm
from utils.api_clients import ModelInterface
from utils.prompts import construct_prompt
from utils.span_processing import parse_labels_from_response, calculate_span_indices
from utils.constants import APIProvider

logging.basicConfig(level=logging.INFO)

def generate_labels_with_reasoner(input_data, provider):
    model = ModelInterface(provider)
    labeled_data = []

    for entry in tqdm(input_data, desc="Processing entries"):
        if not entry.get("model_input") or not entry.get("model_output_text"):
            continue

        prompt = construct_prompt(entry)
        try:
            raw_content = model.generate_completion(prompt)
            labels = parse_labels_from_response(raw_content)

            labeled_entry = {
                "id": entry["id"],
                "lang": entry["lang"],
                "model_input": entry["model_input"],
                "model_output_text": entry["model_output_text"],
                "soft_labels": calculate_span_indices(entry["model_output_text"], labels["soft_labels"]),
                "hard_labels": calculate_span_indices(entry["model_output_text"], labels["hard_labels"]),
            }
            labeled_data.append(labeled_entry)
        except Exception as e:
            logging.error(f"Error processing entry {entry.get('id')}: {e}")

    return labeled_data

def main():
    parser = argparse.ArgumentParser(description="Generate hallucination labels from LLM outputs.")
    parser.add_argument("--lang", required=True, help="Language to process")
    parser.add_argument("--provider", type=str, choices=[p.value for p in APIProvider], default="deepseek")
    args = parser.parse_args()

    lang_map = {
        "catalan": "CA", "arabic": "AR", "finnish": "FI", "farsi": "FA", "basque": "EU",
        "italy": "IT", "swedish": "SV", "czech": "CS", "english": "EN", "german": "DE",
        "spanish": "ES", "chineese": "ZH", "hindi": "HI", "french": "FR"
    }

    input_file = f"unlabeled-test/mushroom.{lang_map[args.lang.lower()]}.jsonl"
    output_file = f"predictions/mushroom-{lang_map[args.lang.lower()]}-predicted.jsonl"

    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]

    labeled_data = generate_labels_with_reasoner(input_data, APIProvider(args.provider))

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in labeled_data:
            json.dump(entry, f)
            f.write("\n")

if __name__ == "__main__":
    main()

