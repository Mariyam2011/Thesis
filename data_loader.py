"""
data_loader.py
--------------
Loads and normalizes GSM8K into the experiment format.
"""

import re
from datasets import load_dataset


# ─────────────────────────────────────────────
# Answer normalization helpers
# ─────────────────────────────────────────────

def normalize_answer(text: str) -> str | None:
    if text is None:
        return None
    text = text.strip().replace(",", "")
    match = re.fullmatch(r"-?\d+(\.\d+)?", text)
    return match.group(0) if match else None


def extract_answer_gsm8k(answer_text: str) -> str | None:
    match = re.search(r"####\s*(-?[\d,]+)", answer_text)
    if match:
        return normalize_answer(match.group(1))
    return None


# ─────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────

def load_gsm8k(split: str = "test") -> list[dict]:
    print(f"[data_loader] Loading GSM8K ({split} split)...")
    dataset = load_dataset("gsm8k", "main", split=split)

    data = []
    skipped = 0
    for item in dataset:
        answer = extract_answer_gsm8k(item["answer"])
        if answer is None:
            skipped += 1
            continue
        data.append({
            "question": item["question"].strip(),
            "answer":   answer,
            "source":   "gsm8k"
        })

    print(f"[data_loader] GSM8K loaded: {len(data)} items ({skipped} skipped)")
    return data

