"""
answer_extraction.py
--------------------
Robust answer extraction from generated text.

Uses a fallback chain — most specific pattern tried first:
  1. Explicit "Final Answer: X" marker  (our prompt asks for this)
  2. LaTeX \\boxed{X}                   (common in math models)
  3. "the answer is X" phrasing
  4. Last number in the output          (last resort only)

This is critical: a broken extractor silently corrupts accuracy AND AUROC.
"""

import re


def normalize_number(text: str) -> str | None:
    """
    Clean a number string:
      - Remove commas: "1,000" → "1000"
      - Strip whitespace
      - Allow negative and decimal
    Returns None if not a valid number.
    """
    if text is None:
        return None
    text = text.strip().replace(",", "").replace(" ", "")
    # Validate: optional minus, digits, optional decimal
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        # Normalize: remove trailing .0
        if text.endswith(".0"):
            text = text[:-2]
        return text
    return None


def extract_final_answer(text: str) -> str | None:
    """
    Extract a numeric answer from generated text using a priority chain.

    Returns a normalized number string, or None if extraction fails.
    """
    if not text or not text.strip():
        return None

    # ── Strategy 1: Explicit marker from our prompt ──
    # Matches "Final Answer: 42" or "Final Answer: -7"
    match = re.search(
        r"[Ff]inal\s+[Aa]nswer\s*[:\s]+(-?[\d,]+(?:\.\d+)?)",
        text
    )
    if match:
        result = normalize_number(match.group(1))
        if result:
            return result

    # ── Strategy 2: LaTeX boxed ──
    # Matches \boxed{42} or \boxed{-7}
    match = re.search(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}", text)
    if match:
        result = normalize_number(match.group(1))
        if result:
            return result

    # ── Strategy 3: Natural language answer pattern ──
    # Matches "the answer is 42", "answer = 42", "equals 42"
    match = re.search(
        r"(?:the\s+answer\s+is|answer\s*=|equals|result\s+is)\s*(-?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )
    if match:
        result = normalize_number(match.group(1))
        if result:
            return result

    # ── Strategy 4: Last number fallback ──
    # Only used as last resort — most likely to be wrong
    all_numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if all_numbers:
        return normalize_number(all_numbers[-1])

    return None


def answers_match(predicted: str | None, ground_truth: str | None) -> bool:
    """
    Compare two answer strings after normalization.
    Handles edge cases: None, whitespace, leading zeros.
    """
    if predicted is None or ground_truth is None:
        return False
    p = normalize_number(predicted)
    g = normalize_number(ground_truth)
    if p is None or g is None:
        return False
    # Try numeric comparison to handle "42" vs "42.0" edge cases
    try:
        return float(p) == float(g)
    except ValueError:
        return p == g