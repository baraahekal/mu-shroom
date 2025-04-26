# MSA at SemEval-2025 Task 3: Multilingual Hallucination Detection

🚀 Official implementation of our SemEval-2025 Task 3 system, designed for detecting hallucinations in multilingual LLM outputs through prompt-engineered weak labeling and ensemble verification.

## Overview

This repository implements our hallucination detection system, which combines:
- **Prompt-based weak label generation** for hallucinated spans.
- **LLM ensemble verification** to validate extracted spans.
- **Post-processing** using fuzzy matching to refine span alignment.

Our system ranked **1st** in Arabic and Basque, and **Top 3** in several other languages at SemEval-2025 Task 3.

---

## 📦 Repository Structure

```
scripts/
│
├── main_label_generator.py    # Main runner script
│
└── utils/
    ├── api_clients.py         # Handles API requests to LLMs
    ├── prompts.py              # Builds task-specific prompts
    ├── span_processing.py      # Parsing and span alignment utilities
    └── constants.py            # API keys management, enums, mappings
```

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:
- `openai`
- `google-generativeai`
- `tqdm`
- `python-dotenv`

---

### 3. Set your API Keys

Create a `.env` file in the root directory (or export manually) with the following content:

```bash
export DEEPSEEK_API_KEY="your-deepseek-key"
export GEMINI_API_KEY="your-gemini-key"
export QWEN_API_KEY="your-qwen-key"
export OPENAI_API_KEY="your-openai-key"
```

Or manually export them before running.

---

## 📋 Usage

### 1. Label data using a selected LLM

```bash
python scripts/main_label_generator.py --lang english --provider openai
```

| Argument | Description |
|:---------|:------------|
| `--lang` | Language to process (e.g., english, arabic, french, etc.) |
| `--provider` | API provider to use (`openai`, `gemini`, `deepseek`, `qwen`) |

The input data should be placed in:
```
unlabeled-test/mushroom.<lang-code>-tst.v1.jsonl
```
and the predictions will be saved into:
```
predictions/mushroom-<lang-code>-tst-predicted.jsonl
```

---

## 📚 Citation

If you use this system, please cite:

```bibtex
@misc{hikal2025msa,
  author = {Baraa Hikal and Ahmed Nasreldin and Ali Hamdi},
  title = {MSA at SemEval-2025 Task 3: Hallucination Detection System},
  year = {2025},
  howpublished = {\url{https://github.com/YourUsername/YourRepoName}},
  note = {Accessed: 2025-04-26}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgements

Special thanks to the organizers of [SemEval-2025 Task 3 (Mu-SHROOM)](https://helsinki-nlp.github.io/shroom/).

---

# ✅ Done

```
Easy to install, easy to run, and fully reproducible 🚀

