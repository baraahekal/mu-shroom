"""
MSA at SemEval-2025 Task 3
Main Hallucination Detection Pipeline
--------------------------------------
Handles span extraction, multi-model adjudication, probability aggregation,
and consensus labeling based on ensemble verification (Section 3.3).
"""

import json
import logging
import argparse
from tqdm import tqdm

from utils.api_clients import ModelInterface
from utils.prompts import construct_prompt
from utils.span_processing import parse_labels_from_response, calculate_span_indices, parse_yes_no_probability
from utils.constants import APIProvider

logging.basicConfig(level=logging.INFO)

# Four models participating in rotation
PROVIDERS = [
    APIProvider.DEEPSEEK,
    APIProvider.GEMINI,
    APIProvider.QWEN,
    APIProvider.OPENAI,
]

def extract_spans(model, input_data):
    """Primary LLM extracts hallucination spans from input."""
    extracted = []
    for entry in tqdm(input_data, desc="Extracting spans"):
        try:
            prompt = construct_prompt(entry)
            raw_content = model.generate_completion(prompt)
            labels = parse_labels_from_response(raw_content)
            spans = calculate_span_indices(entry["model_output_text"], labels["hard_labels"])
            extracted.append({
                "id": entry["id"],
                "lang": entry["lang"],
                "model_input": entry["model_input"],
                "model_output_text": entry["model_output_text"],
                "spans": spans
            })
        except Exception as e:
            logging.error(f"Extraction failed for {entry['id']}: {e}")
    return extracted

def adjudicate_span(span_text, question, adjudicator_models):
    """Each adjudicator model votes (probability) on the hallucination span."""
    votes = []
    for model in adjudicator_models:
        prompt = (
            f"Given the question:\n\n{question}\n\n"
            f"Is the following span hallucinated?\n\nSpan: \"{span_text}\"\n\n"
            f"Answer only a float number between 0.0 (no hallucination) and 1.0 (hallucination)."
        )
        try:
            response = model.generate_completion(prompt)
            prob = parse_yes_no_probability(response)
            votes.append(prob)
        except Exception as e:
            logging.error(f"Adjudication failed: {e}")
            votes.append(0.5)  # Neutral vote
    return votes

def aggregate_votes(votes):
    """Simple average over adjudicator votes."""
    if not votes:
        return 0.0
    return sum(votes) / len(votes)

def label_spans(extracted_spans, adjudicator_models, threshold=0.7):
    """Adjudicate spans and assign final labels."""
    verified = []
    for entry in tqdm(extracted_spans, desc="Adjudicating spans"):
        verified_spans = []
        question = entry["model_input"]

        for span in entry["spans"]:
            span_text = span.get("text")
            votes = adjudicate_span(span_text, question, adjudicator_models)
            avg_prob = aggregate_votes(votes)

            verified_spans.append({
                "text": span_text,
                "start": span.get("start"),
                "end": span.get("end"),
                "avg_prob": avg_prob,
                "hallucinated": avg_prob >= threshold
            })

        verified.append({
            "id": entry["id"],
            "lang": entry["lang"],
            "model_input": entry["model_input"],
            "model_output_text": entry["model_output_text"],
            "verified_spans": verified_spans
        })
    return verified

def save_predictions(predictions, output_path):
    """Save final verified spans."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in predictions:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    logging.info(f"Saved {len(predictions)} examples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run hallucination detection with LLM ensemble verification.")
    parser.add_argument("--lang", type=str, required=True, help="Language to process (e.g., english, arabic, french, etc.)")
    args = parser.parse_args()

    lang_map = {
        "catalan": "CA", "arabic": "AR", "finnish": "FI", "farsi": "FA", "basque": "EU",
        "italy": "IT", "swedish": "SV", "czech": "CS", "english": "EN", "german": "DE",
        "spanish": "ES", "chineese": "ZH", "hindi": "HI", "french": "FR"
    }

    lang_code = lang_map[args.lang.lower()]
    input_path = f"unlabeled-test/mushroom.{lang_code.lower()}-tst.v1.jsonl"

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]

    # Rotate each model as extractor
    for extractor_provider in PROVIDERS:
        adjudicator_providers = [p for p in PROVIDERS if p != extractor_provider]

        logging.info(f"Extractor: {extractor_provider.value}, Adjudicators: {[p.value for p in adjudicator_providers]}")

        extractor_model = ModelInterface(extractor_provider)
        adjudicator_models = [ModelInterface(p) for p in adjudicator_providers]

        # Step 1: Span Extraction
        extracted_spans = extract_spans(extractor_model, input_data)

        # Step 2: Adjudication and Voting
        verified_spans = label_spans(extracted_spans, adjudicator_models)

        # Step 3: Save Results
        output_path = f"predictions/mushroom-{lang_code.lower()}-{extractor_provider.value}-verified.jsonl"
        save_predictions(verified_spans, output_path)

if __name__ == "__main__":
    main()

