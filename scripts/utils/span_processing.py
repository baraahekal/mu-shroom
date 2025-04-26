import re
import json
import logging

def parse_labels_from_response(raw_content: str):
    """Parse response with enhanced JSON extraction and validation."""
    try:
        json_match = re.search(r"```json\s*({.*?})\s*```", raw_content, re.DOTALL)
        if not json_match:
            json_match = re.search(r"\s*({.*?})\s*", raw_content, re.DOTALL)

        if json_match:
            json_str = json_match.group(1).strip()
            parsed = json.loads(json_str)

            if "soft_labels" in parsed:
                parsed["soft_labels"] = [
                    {**item, "start": int(item.get("start", 0)), "end": int(item.get("end", 0))}
                    for item in parsed["soft_labels"] if isinstance(item, dict)
                ]

            return parsed
        else:
            return {"soft_labels": [], "hard_labels": []}
    except Exception as e:
        logging.error(f"JSON parse error: {str(e)}")
        return {"soft_labels": [], "hard_labels": []}

def calculate_span_indices(model_output_text: str, spans):
    updated_spans = []
    for span in spans:
        text = span.get("text") if isinstance(span, dict) else span
        if text:
            start_idx = model_output_text.find(text)
            end_idx = start_idx + len(text) if start_idx != -1 else (-1, -1)
            if isinstance(span, dict):
                updated_spans.append({**span, "start": start_idx, "end": end_idx})
            else:
                updated_spans.append([start_idx, end_idx])
    return updated_spans

