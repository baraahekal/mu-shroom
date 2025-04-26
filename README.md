# MSA at SemEval-2025 Task 3: Multilingual Hallucination Detection

Official implementation of our system for SemEval-2025 Task 3 (Mu-SHROOM), tackling hallucination span detection in multilingual LLM outputs.

🏆 Ranked 1st in Arabic and Basque, Top 3 in several other languages.

---

## Overview

This repository implements a three-stage hallucination detection framework:
- **Span Extraction**: A primary LLM identifies candidate hallucinated spans.
- **Ensemble Adjudication**: Three independent LLMs assign hallucination probabilities per span.
- **Consensus Labeling**: Probabilities are aggregated and thresholded to finalize hallucination labels.

The full process simulates human adjudication and reduces model bias through rotation of extractor and adjudicator roles across four LLMs.

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/baraahekal/mu-shroom.git
   cd mu-shroom
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your API keys:
   ```bash
   export DEEPSEEK_API_KEY="your-key"
   export GEMINI_API_KEY="your-key"
   export QWEN_API_KEY="your-key"
   export OPENAI_API_KEY="your-key"
   ```

---

## Usage

Run hallucination detection with ensemble verification:
```bash
python scripts/main_label_generator.py --lang english
```

- Supported languages: english, arabic, french, german, etc.
- Predictions are saved under the `predictions/` folder, one file per extractor model.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{hikal2025msa,
  author = {Baraa Hikal and Ahmed Nasreldin and Ali Hamdi},
  title = {MSA at SemEval-2025 Task 3: Hallucination Detection System},
  year = {2025},
  howpublished = {\url{https://github.com/baraahekal/mu-shroom}},
  note = {Accessed: 2025-04-26}
}
```

