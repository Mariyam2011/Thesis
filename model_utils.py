"""
model_utils.py
--------------
Model loading and generation with per-token entropy extraction.

Key design decisions:
  1. Entropy is computed per token from raw logits (before sampling).
  2. Tokens are segmented into three types:
       - arithmetic : digits and math operators  → tests your core hypothesis
       - connector  : logical transition words    → baseline comparison
       - other      : everything else
  3. We return both the flat entropy sequence AND per-segment averages.
  4. We store token strings alongside entropies for interpretability.
"""

import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[model_utils] Using device: {DEVICE}")

# ─────────────────────────────────────────────
# Token type definitions
# ─────────────────────────────────────────────

# Tokens that are (or contain) arithmetic content
ARITHMETIC_PATTERN = re.compile(r"[\d\+\-\*\/\=\^\(\)\.]")

# Logical connectors — these are the "hinges" between reasoning steps
CONNECTOR_WORDS = {
    "so", "therefore", "thus", "because", "then", "hence",
    "since", "which", "that", "means", "gives", "yields",
    "resulting", "equals", "is", "are", "have", "get"
}

REPETITION_PENALTY = 1.12
NO_REPEAT_NGRAM_SIZE = 4
MAX_CONSECUTIVE_TOKEN_REPEAT = 25
MAX_FINAL_ANSWER_MARKERS = 2
MAX_IDENTICAL_TRAILING_LINES = 3


def _is_repeated_block_tail(ids: list[int], block_size: int, repeats: int) -> bool:
    """
    Return True if the tail of ids is the same block repeated.
    Example: [..., A,B,A,B,A,B] with block_size=2 and repeats=3.
    """
    n = block_size * repeats
    if len(ids) < n:
        return False
    tail = ids[-n:]
    block = tail[:block_size]
    for i in range(1, repeats):
        if tail[i * block_size:(i + 1) * block_size] != block:
            return False
    return True


class DegenerationStoppingCriteria(StoppingCriteria):
    """
    Stops generation when common loop/degeneration patterns are detected.
    This keeps outputs concise and avoids long repeated templates.
    """

    def __init__(
        self,
        tokenizer,
        prompt_len: int,
        max_consecutive_token_repeat: int,
        max_final_answer_markers: int,
        max_identical_trailing_lines: int,
    ):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.max_consecutive_token_repeat = max_consecutive_token_repeat
        self.max_final_answer_markers = max_final_answer_markers
        self.max_identical_trailing_lines = max_identical_trailing_lines

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0, self.prompt_len:]
        if generated_ids.numel() < 20:
            return False

        ids = generated_ids.tolist()

        # 1) Token-level stutter: same token emitted many times in a row.
        last = ids[-1]
        streak = 0
        for token_id in reversed(ids):
            if token_id == last:
                streak += 1
            else:
                break
        if streak >= self.max_consecutive_token_repeat:
            return True

        # 2) Short repeated block loops (A B A B ... / A B C A B C ...).
        if _is_repeated_block_tail(ids, block_size=2, repeats=6):
            return True
        if _is_repeated_block_tail(ids, block_size=3, repeats=5):
            return True

        # 3) Text-level repeated "Final Answer" scaffolds / line loops.
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if text.lower().count("final answer") >= self.max_final_answer_markers:
            return True

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= self.max_identical_trailing_lines:
            tail = lines[-self.max_identical_trailing_lines:]
            if len(set(tail)) == 1:
                return True

        return False


def classify_token(token_str: str) -> str:
    """
    Return 'arithmetic', 'connector', or 'other' for a decoded token string.
    """
    clean = token_str.strip().lower()
    if ARITHMETIC_PATTERN.search(clean):
        return "arithmetic"
    if clean in CONNECTOR_WORDS:
        return "connector"
    return "other"


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Load tokenizer and model. Uses float16 on GPU, float32 on CPU.
    """
    print(f"[model_utils] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    print(f"[model_utils] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model


# ─────────────────────────────────────────────
# Entropy computation
# ─────────────────────────────────────────────

def compute_entropy(logits_1d: torch.Tensor) -> float:
    """
    Compute Shannon entropy (in nats) from a 1D logit vector.
    H = -sum(p * log(p))
    The 1e-9 epsilon prevents log(0).
    """
    probs = F.softmax(logits_1d, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9))
    return entropy.item()


# ─────────────────────────────────────────────
# Generation with entropy extraction
# ─────────────────────────────────────────────

def generate_with_entropy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    repetition_penalty: float = REPETITION_PENALTY,
    no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
    max_consecutive_token_repeat: int = MAX_CONSECUTIVE_TOKEN_REPEAT,
    max_final_answer_markers: int = MAX_FINAL_ANSWER_MARKERS,
    max_identical_trailing_lines: int = MAX_IDENTICAL_TRAILING_LINES,
) -> dict:
    """
    Generate a response and extract per-token entropy.

    Returns a dict with:
        generated_text   : str         — full decoded output
        prompt_text      : str         — original prompt (for reference)
        tokens           : list[str]   — decoded token strings (new tokens only)
        entropies        : list[float] — per-token entropy values
        token_types      : list[str]   — 'arithmetic' | 'connector' | 'other'
        segment_entropy  : dict        — mean entropy per token type
        entropy_mean     : float       — mean over all tokens
        entropy_max      : float       — max over all tokens
        entropy_last_10  : float       — mean over last 10 tokens (pre-answer region)
        entropy_first_10 : float       — mean over first 10 tokens (problem setup)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    stopping_criteria = StoppingCriteriaList(
        [DegenerationStoppingCriteria(
            tokenizer,
            prompt_len=input_len,
            max_consecutive_token_repeat=max_consecutive_token_repeat,
            max_final_answer_markers=max_final_answer_markers,
            max_identical_trailing_lines=max_identical_trailing_lines,
        )]
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,          # get logits at each step
            do_sample=False,             # greedy — deterministic, reproducible
            temperature=1.0,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    # outputs.scores: tuple of (vocab_size,) tensors, one per generated token
    # outputs.sequences: (1, input_len + new_tokens)
    generated_ids = outputs.sequences[0][input_len:]   # new tokens only
    scores = outputs.scores                             # logits per step

    # ── Per-token entropy and classification ──
    entropies    = []
    token_types  = []
    token_strs   = []

    for token_id, step_logits in zip(generated_ids, scores):
        token_str   = tokenizer.decode([token_id.item()])
        token_type  = classify_token(token_str)
        entropy_val = compute_entropy(step_logits[0])   # step_logits shape: (1, vocab)

        token_strs.append(token_str)
        token_types.append(token_type)
        entropies.append(entropy_val)

    # ── Segment-level entropy ──
    segment_buckets: dict[str, list[float]] = {
        "arithmetic": [], "connector": [], "other": []
    }
    for e, t in zip(entropies, token_types):
        segment_buckets[t].append(e)

    segment_entropy = {
        k: (sum(v) / len(v) if v else None)
        for k, v in segment_buckets.items()
    }

    # ── Aggregate stats ──
    n = len(entropies)
    entropy_mean     = sum(entropies) / n if n > 0 else 0.0
    entropy_max      = max(entropies)      if n > 0 else 0.0
    entropy_last_10  = sum(entropies[-10:]) / min(10, n) if n > 0 else 0.0
    entropy_first_10 = sum(entropies[:10])  / min(10, n) if n > 0 else 0.0

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "generated_text":   generated_text,
        "prompt_text":      prompt,
        "tokens":           token_strs,
        "entropies":        entropies,
        "token_types":      token_types,
        "segment_entropy":  segment_entropy,
        "entropy_mean":     entropy_mean,
        "entropy_max":      entropy_max,
        "entropy_last_10":  entropy_last_10,
        "entropy_first_10": entropy_first_10,
    }