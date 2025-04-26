import json, time
import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from openai import OpenAI
from google import genai


class APIProvider(Enum):
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    QWEN = "qwen"
    OPENAI = "openai"

@dataclass
class APIConfig:
    api_key: str
    base_url: Optional[str] = None
    model_name: str = ""
    http_options: Optional[Dict[str, Any]] = None

API_CONFIGS = {
    APIProvider.DEEPSEEK: APIConfig(
        api_key="sk-b5e3d5d928fd4ef8a1317c93ba1c85ea",
        base_url="https://api.deepseek.com",
        model_name="deepseek-chat"
    ),
    APIProvider.GEMINI: APIConfig(
        api_key="AIzaSyApCtQbQ1K_eM6PVUzPrmPfiNHa0mWtomg",
        model_name="gemini-2.0-flash-thinking-exp",
        http_options={'api_version': 'v1alpha'}
    ),
    APIProvider.QWEN: APIConfig(
        api_key="sk-353054e9177244de94943786dd8cadb4",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus"
    ),
    APIProvider.OPENAI: APIConfig(
        api_key="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",
        model_name="gpt-4o"
    )
}

LANGS = {
    "AR": "ARABIC",
    "FI": "FINNISH",
    "FA": "FARSI",
    "EU": "BASQUE",
    "IT": "ITALIANO",
    "SV": "SWEDISH",
    "CA": "CATALAN",
    "CS": "CZECH",
    "EN": "ENGLISH",
    "DE": "GERMAN",
    "ES": "SPANISH",
    "ZH": "CHINEESE",
    "HI": "HINDI",
    "FR": "FRENCH"
}

class APIClientFactory:
    @staticmethod
    def create_client(provider: APIProvider) -> Any:
        config = API_CONFIGS[provider]

        if provider in [APIProvider.DEEPSEEK, APIProvider.QWEN]:
            return OpenAI(api_key=config.api_key, base_url=config.base_url)
        elif provider == APIProvider.GEMINI:
            return genai.Client(api_key=config.api_key, http_options=config.http_options)
        elif provider == APIProvider.OPENAI:
            return OpenAI(api_key=config.api_key)
        else:
            raise ValueError(f"Unsupported API provider: {provider}")

class ModelInterface:
    def __init__(self, provider: APIProvider):
        self.provider = provider
        self.client = APIClientFactory.create_client(provider)
        self.config = API_CONFIGS[provider]

    def generate_completion(self, prompt: str) -> str:
        try:
            if self.provider in [APIProvider.DEEPSEEK, APIProvider.QWEN]:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
                return response.choices[0].message.content

            elif self.provider == APIProvider.GEMINI:
                config = {'thinking_config': {'include_thoughts': True}}
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=config
                )
                return response.text

            elif self.provider == APIProvider.OPENAI:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
                return response.choices[0].message.content

        except Exception as e:
            logging.error(f"Error generating completion with {self.provider}: {str(e)}")
            raise

def parse_labels_from_response(raw_content: str) -> Dict[str, List]:
    """Parse response with enhanced JSON extraction and validation"""
    try:
        # Extract JSON block from markdown code fences
        json_match = re.search(r"```json\s*({.*?})\s*```", raw_content, re.DOTALL)
        if not json_match:
            # Fallback: find first JSON object
            json_match = re.search(r"\s*({.*?})\s*", raw_content, re.DOTALL)

        if json_match:
            json_str = json_match.group(1).strip()
            parsed = json.loads(json_str)

            # Only convert 'start'/'end' for soft_labels
            if "soft_labels" in parsed:
                new_soft_labels = []
                for item in parsed["soft_labels"]:
                    if isinstance(item, dict):
                        for k in ["start", "end"]:
                            if k in item:
                                item[k] = int(item[k])
                        new_soft_labels.append(item)
                    else:
                        new_soft_labels.append(item)
                parsed["soft_labels"] = new_soft_labels

            return parsed
        else:
            return {"soft_labels": [], "hard_labels": []}
    except Exception as e:
        logging.error(f"JSON parse error: {str(e)}")
        return {"soft_labels": [], "hard_labels": []}

def calculate_span_indices(model_output_text: str, spans: List) -> List:
    """
    Adds 'start' and 'end' indices to spans based on model_output_text.
    """
    updated_spans = []
    for span in spans:
        if isinstance(span, dict):  # For soft_labels
            text = span.get("text")
        else:  # For hard_labels
            text = span

        if text:
            start_idx = model_output_text.find(text)
            if start_idx != -1:
                end_idx = start_idx + len(text)
                if isinstance(span, dict):
                    updated_spans.append({
                        "text": text,
                        "start": start_idx,
                        "end": end_idx,
                        "prob": span.get("prob")
                    })
                else:
                    updated_spans.append([start_idx, end_idx])
            else:
                if isinstance(span, dict):
                    updated_spans.append({"text": text, "start": -1, "end": -1})
                else:
                    updated_spans.append([-1, -1])
    return updated_spans

def construct_prompt(entry: Dict[str, Any]) -> str:
    """Constructs the prompt for the model"""
    return (
        f"### Question & Answer Pair\n"
        f"**Question:** {entry['model_input']}\n"
        f"**Answer:** {entry['model_output_text']}\n\n"

        f"### Task Description\n"
        f"You are a professional annotator and {LANGS[entry['lang']]} linguistic expert. Your job is to detect and extract **hallucination spans** from the provided answer compared to the question. "

        f"### Annotation Principles\n"
        f"1. **Exact Span Matching**:\n"
        f"   - Extract spans *word-for-word* and *character-for-character* exactly as they appear in the answer.\n"
        f"   - Ensure perfect alignment, including punctuation, capitalization, and spacing.\n"
        f"   - If a span is partially supported, only extract the **unsupported portion**.\n\n"
        f"   - **Preserve original numeral formats**:\n"
        f"     - **Persian/Arabic numerals must remain in their native script** (e.g., '۱۹۷٤' should not be converted to '1974').\n"

        f"2. **Minimal Spans**:\n"
        f"   - Select the **smallest possible** spans that, when removed, completely eliminate the hallucination.\n"
        f"   - **Prioritize precision:** Avoid extracting entire sentences if a **shorter phrase** accurately captures the hallucination.\n"
        f"   - Ensure the extracted span **exclusively** contains hallucinated content without removing valid information.\n\n"

        f"3. **Hallucination Definition**:\n"
        f"   - Any phrase, entity, number, or fact that is **not supported** by the question.\n"
        f"   - Any **exaggeration** or **overly specific** detail absent in the question.\n"
        f"   - Incorrect **names, locations, numbers, dates, or causes**.\n"
        f"   - In yes/no questions, unsupported answers (e.g., 'Yes', 'No') and speculative details.\n\n"

        f"4. **Soft and Hard Labels**:\n"
        f"   - Assign **probabilities (0.0 - 1.0)** for soft labels based on hallucination confidence.\n"
        f"   - Include spans with **≥ 0.7 probability** in hard labels.\n\n"

        f"### Example 1:\n"
        f"**Q:** \"كم عدد المقاطعات في جليقية؟\"\n"
        f"**A:** \"جليقة، التي تعرف الآن باسم كوريا الجنوبية، تتألف من 16 مقاطعة.\"\n"
        f"```json\n"
        f"{{\n"
        f"  \"soft_labels\": [\n"
        f"    {{\"text\": \"كوريا الجنوبية\", \"prob\": 1.0}},\n"
        f"    {{\"text\": \"16\", \"prob\": 1.0}},\n"
        f"    {{\"text\": \" مقاطعة\", \"prob\": 0.33}}\n"
        f"  ],\n"
        f"  \"hard_labels\": [\"كوريا الجنوبية\", \"16\"]\n"
        f"}}\n"
        f"```\n\n"

        f"### Example 2:\n"
        f"**Q:** \"هل شارك ديف فرانكو في فيلم ليغو؟\"\n"
        f"**A:** \"نعم، شارک داف فرانك في فيلم ليجو (2017) في دور (هالوسينغ هيلبي).\"\n"
        f"```json\n"
        f"{{\n"
        f"  \"soft_labels\": [\n"
        f"    {{\"text\": \"نعم، شارک\", \"prob\": 0.33}},\n"
        f"    {{\"text\": \"2017\", \"prob\": 1.0}},\n"
        f"    {{\"text\": \"هالوسينغ هيلبي\", \"prob\": 0.67}}\n"
        f"  ],\n"
        f"  \"hard_labels\": [\"2017\", \"هالوسينغ هيلبي\"]\n"
        f"}}\n"
        f"```\n\n"

        f"### Example 3:\n"
        f"**Q:** \"ما يميز ملعب والز سيزينهيم عن باقي الملاعب في النمسا؟\"\n"
        f"**A:** \"ملعب وولز سيغنهايم هو ملعب هوكي في نمسا، النرويج. يعتبر من بين أكبر الملاجئ في أوروبا، مع استيعاب 15.005 مشجع.\"\n"
        f"```json\n"
        f"{{\n"
        f"  \"soft_labels\": [\n"
        f"    {{\"text\": \"ملعب\", \"prob\": 0.67}},\n"
        f"    {{\"text\": \"هوكي\", \"prob\": 1.0}},\n"
        f"    {{\"text\": \"النرويج\", \"prob\": 0.33}},\n"
        f"    {{\"text\": \"15.005\", \"prob\": 1.0}}\n"
        f"  ],\n"
        f"  \"hard_labels\": [\"هوكي\", \"15.005\"]\n"
        f"}}\n"
        f"```\n\n"

        f"### Additional Guidelines:\n"
        f"- **For numerical hallucinations**, extract **only** the incorrect number.\n"
        f"- **For location or entity errors**, extract **just the wrong portion**.\n"
        f"- If **no hallucination** is found, return an empty list `\"soft_labels\": []`.\n"
        f"- Ensure that **every extracted span is present verbatim** in the hypothesis.\n\n"

        f"### Output Format\n"
        f"```json\n"
        f"{{\n"
        f"  \"soft_labels\": [\n"
        f"    {{\"text\": \"<span_text>\", \"prob\": <float>}},\n"
        f"    ...\n"
        f"  ],\n"
        f"  \"hard_labels\": [\"<span_text>\", ...]\n"
        f"}}\n"
        f"```\n"

        f"Your task: Extract hallucination spans **precisely** from the answer and return them in JSON format."
    )



def generate_labels_with_reasoner(input_data: List[Dict], provider: APIProvider = APIProvider.DEEPSEEK) -> List[Dict]:
    """
    Generates soft and hard labels using the specified model with a progress bar.
    """
    labeled_data = []
    ct = 0

    model = ModelInterface(provider)

    for entry in tqdm(input_data, desc="Processing entries"):
        if not entry["model_input"] or not entry["model_output_text"]:
            logging.warning(f"Skipping empty input for entry {entry['id']}")
            continue

        prompt = construct_prompt(entry)

        try:
            raw_content = model.generate_completion(prompt)
            # print(raw_content)
            labels = parse_labels_from_response(raw_content)

            labeled_entry = {
                "id": entry["id"],
                "lang": entry["lang"],
                "model_input": entry['model_input'],
                "model_output_text": entry['model_output_text'],
                "soft_labels": calculate_span_indices(entry["model_output_text"], labels["soft_labels"]),
                "hard_labels": calculate_span_indices(entry["model_output_text"], labels["hard_labels"]),
            }
            labeled_data.append(labeled_entry)

            if provider == APIProvider.GEMINI and ct % 10 == 0 and ct > 1:
                logging.info("Rate limiting pause...")
                time.sleep(50)
            elif provider == APIProvider.DEEPSEEK and ct % 50 == 0 and ct > 1:
                logging.info("Rate limiting pause...")
            ct += 1

        except Exception as e:
            logging.error(f"Error processing entry {entry.get('id')}: {str(e)}")

    return labeled_data

def label_and_save_data(file_path: str, output_path: str, provider: APIProvider):
    """
    Load unlabeled data, generate labels, and save the labeled dataset.
    """
    logging.info(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]

    logging.info(f"Generating labels using {provider.value}...")
    labeled_data = generate_labels_with_reasoner(input_data, provider)

    logging.info(f"Saving labeled data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in labeled_data:
            json.dump(entry, f)
            f.write("\n")

    logging.info("Labeling complete.")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate labels using various API providers.")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["catalan", "arabic", "finnish", "farsi", "basque", "italy", "swedish", "czech", "english", "german", "spanish", "chineese", "hindi", "french"],
        help="Language to process"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=[p.value for p in APIProvider],
        default=APIProvider.DEEPSEEK.value,
        help="API provider to use"
    )

    args = parser.parse_args()
    provider = APIProvider(args.provider)

    lang_map = {
        "catalan": "CA",
        "arabic": "AR",
        "finnish": "FI",
        "farsi": "FA",
        "basque": "EU",
        "italy": "IT",
        "swedish": "SV",
        "czech": "CS",
        "english": "EN",
        "german": "DE",
        "spanish": "ES",
        "chinese": "ZH",
        "hindi": "HI",
        "french": "FR"
    }

    lang_code = lang_map[args.lang.lower()]
    input_file = f"unlabeled-test/mushroom.{lang_code.lower()}-tst.v1.jsonl"
    output_file = f"predictions/mushroom-{lang_code.lower()}-tst-predicted.jsonl"

    label_and_save_data(input_file, output_file, provider)

if __name__ == "__main__":
    main()
