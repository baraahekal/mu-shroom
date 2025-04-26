from .constants import LANGS

def construct_prompt(entry):
    return (
        f"### Question & Answer Pair\n"
        f"**Question:** {entry['model_input']}\n"
        f"**Answer:** {entry['model_output_text']}\n\n"
        f"### Task Description\n"
        f"You are a professional annotator and {LANGS[entry['lang']]} linguistic expert. "
        f"Your job is to detect hallucination spans compared to the question.\n\n"
        f"[Full prompt continues here... depending on your original version]"
    )

