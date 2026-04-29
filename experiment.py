"""
experiment.py
-------------
Runs inference on a dataset and saves results incrementally.

Key features:
  - Saves after every N problems (crash-safe)
  - Skips already-completed problems (resumable)
  - Records full entropy metadata per problem
  - Supports multiple prompt strategies
"""

import json
import time
from pathlib import Path

from answer_extraction import extract_final_answer, answers_match
from model_utils import generate_with_entropy
from prompts import cot_prompt, pot_prompt, direct_prompt


PROMPT_FNS = {
    "cot":    cot_prompt,
    "pot":    pot_prompt,
    "direct": direct_prompt,
}


def run_experiment(
    dataset:    list[dict],
    tokenizer,
    model,
    strategy:   str = "cot",           # "cot" | "pot" | "direct"
    output_path: str = "results/run.json",
    save_every:  int = 10,             # save checkpoint every N problems
    max_new_tokens: int = 512,
    limit:       int | None = None,    # set to e.g. 200 for quick runs
) -> list[dict]:
    """
    Run inference over dataset and return list of result dicts.

    Each result contains:
        question, ground_truth, prediction, correct,
        strategy, source, generated_text,
        entropy_mean, entropy_max, entropy_last_10, entropy_first_10,
        segment_entropy (per token type),
        entropies (full sequence),
        token_types (full sequence),
        tokens (full sequence),
        inference_time_s
    """
    assert strategy in PROMPT_FNS, f"Unknown strategy '{strategy}'. Choose from {list(PROMPT_FNS)}"
    prompt_fn = PROMPT_FNS[strategy]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint if exists ──
    results = []
    completed_questions = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            results = json.load(f)
        completed_questions = {r["question"] for r in results}
        print(f"[experiment] Resuming: {len(results)} already completed.")

    # ── Subset if requested ──
    data = dataset[:limit] if limit else dataset
    remaining = [item for item in data if item["question"] not in completed_questions]
    print(f"[experiment] Running {len(remaining)} problems (strategy={strategy})")

    for i, item in enumerate(remaining):
        prompt = prompt_fn(item["question"])

        t0 = time.time()
        try:
            gen = generate_with_entropy(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens
            )
        except Exception as e:
            print(f"[experiment] ERROR on problem {i}: {e}")
            continue
        elapsed = time.time() - t0

        prediction   = extract_final_answer(gen["generated_text"])
        is_correct   = answers_match(prediction, item["answer"])

        result = {
            # ── Identity ──
            "question":       item["question"],
            "ground_truth":   item["answer"],
            "source":         item.get("source", "unknown"),
            "strategy":       strategy,

            # ── Outcome ──
            "prediction":     prediction,
            "correct":        is_correct,
            "generated_text": gen["generated_text"],

            # ── Entropy aggregates ──
            "entropy_mean":      gen["entropy_mean"],
            "entropy_max":       gen["entropy_max"],
            "entropy_last_10":   gen["entropy_last_10"],
            "entropy_first_10":  gen["entropy_first_10"],
            "segment_entropy":   gen["segment_entropy"],   # per token type

            # ── Full sequences (for post-hoc analysis) ──
            "entropies":    gen["entropies"],
            "token_types":  gen["token_types"],
            "tokens":       gen["tokens"],

            # ── Metadata ──
            "inference_time_s": round(elapsed, 3),
        }

        results.append(result)

        # ── Progress print ──
        status = "✓" if is_correct else "✗"
        print(
            f"  [{i+1:4d}/{len(remaining)}] {status} | "
            f"pred={prediction} gt={item['answer']} | "
            f"H_mean={gen['entropy_mean']:.3f} "
            f"H_max={gen['entropy_max']:.3f} | "
            f"{elapsed:.1f}s"
        )

        # ── Incremental save ──
        if (i + 1) % save_every == 0:
            _save(results, output_path)
            print(f"  [checkpoint saved → {output_path}]")

    # ── Final save ──
    _save(results, output_path)
    print(f"[experiment] Done. Results saved to {output_path}")
    return results


def _save(results: list[dict], path: Path):
    """Save results to JSON, excluding full token sequences to keep file small."""
    # Save a compact version (no full token sequences — those are large)
    compact = []
    for r in results:
        c = {k: v for k, v in r.items() if k not in ("tokens", "token_types", "entropies")}
        compact.append(c)

    with open(path, "w") as f:
        json.dump(compact, f, indent=2)

    # Also save full version separately
    full_path = path.with_stem(path.stem + "_full")
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)