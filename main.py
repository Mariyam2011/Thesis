"""
main.py
-------
Entry point for the entropy-based failure prediction thesis experiment.

Pipeline:
  1. Load datasets (GSM8K by default)
  2. Load model (Qwen2.5-0.5B-Instruct by default)
  3. Run inference with entropy extraction across prompt strategies
  4. Analyze results and generate report

Usage examples:
  # Quick smoke test (20 GSM8K problems, CoT only)
  python main.py --limit 20 --strategies cot

  # Full run, all strategies
  python main.py

  # Run using a LoRA fine-tuned adapter (r=16)
  python main.py --model Qwen/Qwen3-0.6B --adapter-path lora_outputs/qwen3_gsm8k/adapter --strategies cot

  # LoRA rank ablation (train r=32/64, eval vs r=16 baseline)
  python run_lora_rank_experiment.py --ranks 32 64 --include-baseline

  # Resume an interrupted run
  python main.py --resume --strategies cot

  # Only analyze existing results (skip inference)
  python main.py --analyze-only --results-dir results/
"""

import argparse
import json
import os
from pathlib import Path

from data_loader import load_gsm8k
from model_utils import load_model
from experiment import run_experiment
from analysis import generate_report


# ─────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────

DEFAULT_MODEL      = "Qwen/Qwen3-0.6B"
DEFAULT_STRATEGIES = ["cot", "pot", "direct"]
DEFAULT_DATASETS    = ["gsm8k"]
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_MAX_TOKENS  = 512
DEFAULT_SAVE_EVERY  = 10
DEFAULT_REPETITION_PENALTY = 1.12
DEFAULT_NO_REPEAT_NGRAM_SIZE = 4
DEFAULT_MAX_CONSECUTIVE_TOKEN_REPEAT = 25
DEFAULT_MAX_FINAL_ANSWER_MARKERS = 2
DEFAULT_MAX_IDENTICAL_TRAILING_LINES = 3


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entropy-based failure prediction for small LLMs on math benchmarks."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
        help="Optional LoRA adapter directory (e.g., lora_outputs/.../adapter) to evaluate fine-tuned model."
    )
    parser.add_argument(
        "--strategies", nargs="+", default=DEFAULT_STRATEGIES,
        choices=["cot", "pot", "direct"],
        help="Prompt strategies to run (default: all three)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max problems per dataset per strategy (None = full). Use 20–50 for smoke tests."
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Directory to save results and reports"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Max new tokens per generation (default: 512)"
    )
    parser.add_argument(
        "--save-every", type=int, default=DEFAULT_SAVE_EVERY,
        help="Checkpoint save frequency in problems (default: 10)"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY,
        help="Generation repetition penalty (>1 discourages repeating tokens)."
    )
    parser.add_argument(
        "--no-repeat-ngram-size", type=int, default=DEFAULT_NO_REPEAT_NGRAM_SIZE,
        help="Block repeated n-grams of this size during generation."
    )
    parser.add_argument(
        "--max-consecutive-token-repeat", type=int, default=DEFAULT_MAX_CONSECUTIVE_TOKEN_REPEAT,
        help="Stop if one token repeats this many times consecutively."
    )
    parser.add_argument(
        "--max-final-answer-markers", type=int, default=DEFAULT_MAX_FINAL_ANSWER_MARKERS,
        help="Stop if 'Final Answer' appears this many times."
    )
    parser.add_argument(
        "--max-identical-trailing-lines", type=int, default=DEFAULT_MAX_IDENTICAL_TRAILING_LINES,
        help="Stop if the last N non-empty lines are identical."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        choices=["gsm8k"],
        help="Datasets to run on (default: gsm8k only)"
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Skip inference; load existing results and regenerate report only"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoints (skips already-completed problems)"
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Ignore existing result files and run inference from scratch."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────

def load_datasets(dataset_names: list[str]) -> dict[str, list[dict]]:
    """Load requested datasets and return as a dict keyed by name."""
    datasets = {}
    if "gsm8k" in dataset_names:
        datasets["gsm8k"] = load_gsm8k(split="test")
    return datasets


# ─────────────────────────────────────────────
# Analysis-only mode
# ─────────────────────────────────────────────

def load_all_results(results_dir: Path, dataset_names: list[str] | None = None) -> list[dict]:
    """
    Load selected *_full.json result files from the results directory
    and merge them into a single list for combined analysis.
    """
    all_results = []
    full_files = list(results_dir.glob("*_full.json"))

    if dataset_names:
        dataset_set = set(dataset_names)
        full_files = [
            path for path in full_files
            if any(f"_{dataset_name}_full" in path.stem for dataset_name in dataset_set)
        ]

    if not full_files:
        # Fall back to compact files if no full files found
        full_files = list(results_dir.glob("*.json"))
        full_files = [f for f in full_files if "report" not in f.name]
        if dataset_names:
            dataset_set = set(dataset_names)
            full_files = [
                path for path in full_files
                if any(path.stem.endswith(f"_{dataset_name}") for dataset_name in dataset_set)
            ]

    if not full_files:
        print(f"[main] No result files found in {results_dir}")
        return []

    for path in sorted(full_files):
        print(f"[main] Loading results from {path}...")
        with open(path, "r") as f:
            data = json.load(f)
        all_results.extend(data)
        print(f"  → {len(data)} problems loaded")

    print(f"[main] Total problems loaded: {len(all_results)}")
    return all_results


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ENTROPY THESIS EXPERIMENT")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Adapter:    {args.adapter_path or 'none'}")
    print(f"  Strategies: {args.strategies}")
    print(f"  Datasets:   {args.datasets}")
    print(f"  Limit:      {args.limit or 'full'}")
    print(f"  rep_penalty={args.repetition_penalty} ngram={args.no_repeat_ngram_size} "
          f"repeat_token={args.max_consecutive_token_repeat} "
          f"final_answer={args.max_final_answer_markers} "
          f"line_repeat={args.max_identical_trailing_lines}")
    print(f"  Results →   {results_dir}")
    print("=" * 60 + "\n")

    # ── Analyze-only mode ──────────────────────────────────────
    if args.analyze_only:
        print("[main] Analyze-only mode — skipping inference.")
        all_results = load_all_results(results_dir, args.datasets)
        if not all_results:
            print("[main] No results to analyze. Run inference first.")
            return
        report_path = results_dir / "report_combined.json"
        generate_report(all_results, output_path=str(report_path))
        return

    # ── Load datasets ──────────────────────────────────────────
    print("[main] Loading datasets...")
    datasets = load_datasets(args.datasets)
    if not datasets:
        print("[main] ERROR: No datasets loaded. Exiting.")
        return
    for name, data in datasets.items():
        print(f"  {name}: {len(data)} problems")

    # ── Load model ─────────────────────────────────────────────
    print(f"\n[main] Loading model: {args.model}")
    tokenizer, model = load_model(args.model, adapter_path=args.adapter_path)

    # ── Run experiments ────────────────────────────────────────
    all_results = []

    for strategy in args.strategies:
        for dataset_name, dataset in datasets.items():
            run_label = f"{strategy}_{dataset_name}"
            output_path = results_dir / f"{run_label}.json"

            print(f"\n{'─'*50}")
            print(f"[main] Strategy={strategy} | Dataset={dataset_name}")
            print(f"{'─'*50}")

            # If not resuming and file exists, either skip or force rerun.
            if not args.resume and not args.force_rerun and output_path.exists():
                print(f"[main] Results already exist at {output_path}.")
                print(f"       Use --resume to continue, --force-rerun to overwrite, or delete the file to re-run.")

                # For combined analysis, prefer full results (with token-level data)
                # so spike analysis is not silently zeroed.
                full_existing_path = output_path.with_stem(output_path.stem + "_full")
                load_path = full_existing_path if full_existing_path.exists() else output_path
                with open(load_path, "r") as f:
                    existing = json.load(f)
                all_results.extend(existing)
                continue

            results = run_experiment(
                dataset=dataset,
                tokenizer=tokenizer,
                model=model,
                strategy=strategy,
                output_path=str(output_path),
                save_every=args.save_every,
                max_new_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_consecutive_token_repeat=args.max_consecutive_token_repeat,
                max_final_answer_markers=args.max_final_answer_markers,
                max_identical_trailing_lines=args.max_identical_trailing_lines,
                limit=args.limit,
            )
            all_results.extend(results)

    # ── Combined report ────────────────────────────────────────
    if all_results:
        print(f"\n[main] Generating combined report over {len(all_results)} results...")
        report_path = results_dir / "report_combined.json"
        generate_report(all_results, output_path=str(report_path))
    else:
        print("[main] No results collected — nothing to report.")

    # ── Per-strategy/dataset reports ──────────────────────────
    print("\n[main] Generating per-run reports...")
    for strategy in args.strategies:
        for dataset_name in datasets:
            run_label = f"{strategy}_{dataset_name}"
            full_path = results_dir / f"{run_label}_full.json"
            report_path = results_dir / f"report_{run_label}.json"

            if full_path.exists():
                with open(full_path, "r") as f:
                    run_results = json.load(f)
                if run_results:
                    generate_report(run_results, output_path=str(report_path))
            else:
                # Try compact file
                compact_path = results_dir / f"{run_label}.json"
                if compact_path.exists():
                    with open(compact_path, "r") as f:
                        run_results = json.load(f)
                    if run_results:
                        generate_report(run_results, output_path=str(report_path))

    print("\n[main] All done!")
    print(f"  Results saved in: {results_dir.resolve()}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()