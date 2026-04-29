"""
data_loader.py
--------------
Loads and normalizes GSM8K and MATH500 datasets into a unified format.
"""

import re
import json
from pathlib import Path
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


def extract_answer_math500(answer_text: str) -> str | None:
    match = re.search(r"\\boxed\{([^}]+)\}", answer_text)
    if match:
        return normalize_answer(match.group(1))
    return normalize_answer(answer_text)


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


def load_math500(path: str = "data/math500.json") -> list[dict]:
    path = Path(path)
    if not path.exists():
        print("[data_loader] math500.json not found, attempting HuggingFace load...")
        return _load_math500_hf()

    print(f"[data_loader] Loading MATH500 from {path}...")
    with open(path, "r") as f:
        raw = json.load(f)

    data = []
    skipped = 0
    for item in raw:
        answer = extract_answer_math500(item.get("answer", item.get("solution", "")))
        if answer is None:
            skipped += 1
            continue
        data.append({
            "question": item["problem"].strip(),
            "answer":   answer,
            "source":   "math500"
        })

    print(f"[data_loader] MATH500 loaded: {len(data)} items ({skipped} skipped)")
    return data


def _load_math500_hf() -> list[dict]:
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        data = []
        for item in dataset:
            # This dataset has a direct 'answer' field
            answer = normalize_answer(item.get("answer", ""))
            if answer is None:
                answer = extract_answer_math500(item.get("solution", ""))
            if answer is None:
                continue
            data.append({
                "question": item["problem"].strip(),
                "answer":   answer,
                "source":   "math500"
            })
        print(f"[data_loader] MATH500 (HF) loaded: {len(data)} items")
        return data
    except Exception as e:
        print(f"[data_loader] ERROR loading MATH500 from HuggingFace: {e}")
        return []