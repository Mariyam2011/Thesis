"""
main.py
-------
Entry point for the entropy-based failure prediction thesis experiment.

Pipeline:
  1. Load datasets (GSM8K + MATH500)
  2. Load model (Qwen2.5-0.5B-Instruct by default)
  3. Run inference with entropy extraction across prompt strategies
  4. Analyze results and generate report

Usage examples:
  # Quick smoke test (20 problems, CoT only)
  python main.py --limit 20 --strategies cot

  # Full run, all strategies
  python main.py

  # Resume an interrupted run
  python main.py --resume --strategies cot

  # Only analyze existing results (skip inference)
  python main.py --analyze-only --results-dir results/
"""

import argparse
import json
import os
from pathlib import Path

from data_loader import load_gsm8k, load_math500
from model_utils import load_model
from experiment import run_experiment
from analysis import generate_report


# ─────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────

DEFAULT_MODEL      = "Qwen/Qwen3-0.6B"
DEFAULT_STRATEGIES = ["cot", "pot", "direct"]
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_MAX_TOKENS  = 512
DEFAULT_SAVE_EVERY  = 10


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
        "--datasets", nargs="+", default=["gsm8k", "math500"],
        choices=["gsm8k", "math500"],
        help="Datasets to run on (default: both)"
    )
    parser.add_argument(
        "--math500-path", type=str, default="data/math500.json",
        help="Local path to math500.json (will fall back to HuggingFace if missing)"
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Skip inference; load existing results and regenerate report only"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoints (skips already-completed problems)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────

def load_datasets(dataset_names: list[str], math500_path: str) -> dict[str, list[dict]]:
    """Load requested datasets and return as a dict keyed by name."""
    datasets = {}
    if "gsm8k" in dataset_names:
        datasets["gsm8k"] = load_gsm8k(split="test")
    if "math500" in dataset_names:
        datasets["math500"] = load_math500(path=math500_path)
    return datasets


# ─────────────────────────────────────────────
# Analysis-only mode
# ─────────────────────────────────────────────

def load_all_results(results_dir: Path) -> list[dict]:
    """
    Load all *_full.json result files from the results directory
    and merge them into a single list for combined analysis.
    """
    all_results = []
    full_files = list(results_dir.glob("*_full.json"))

    if not full_files:
        # Fall back to compact files if no full files found
        full_files = list(results_dir.glob("*.json"))
        full_files = [f for f in full_files if "report" not in f.name]

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
    print(f"  Strategies: {args.strategies}")
    print(f"  Datasets:   {args.datasets}")
    print(f"  Limit:      {args.limit or 'full'}")
    print(f"  Results →   {results_dir}")
    print("=" * 60 + "\n")

    # ── Analyze-only mode ──────────────────────────────────────
    if args.analyze_only:
        print("[main] Analyze-only mode — skipping inference.")
        all_results = load_all_results(results_dir)
        if not all_results:
            print("[main] No results to analyze. Run inference first.")
            return
        report_path = results_dir / "report_combined.json"
        generate_report(all_results, output_path=str(report_path))
        return

    # ── Load datasets ──────────────────────────────────────────
    print("[main] Loading datasets...")
    datasets = load_datasets(args.datasets, args.math500_path)
    if not datasets:
        print("[main] ERROR: No datasets loaded. Exiting.")
        return
    for name, data in datasets.items():
        print(f"  {name}: {len(data)} problems")

    # ── Load model ─────────────────────────────────────────────
    print(f"\n[main] Loading model: {args.model}")
    tokenizer, model = load_model(args.model)

    # ── Run experiments ────────────────────────────────────────
    all_results = []

    for strategy in args.strategies:
        for dataset_name, dataset in datasets.items():
            run_label = f"{strategy}_{dataset_name}"
            output_path = results_dir / f"{run_label}.json"

            print(f"\n{'─'*50}")
            print(f"[main] Strategy={strategy} | Dataset={dataset_name}")
            print(f"{'─'*50}")

            # If resuming and file exists, results will be loaded inside run_experiment
            if not args.resume and output_path.exists():
                print(f"[main] Results already exist at {output_path}.")
                print(f"       Use --resume to continue, or delete the file to re-run.")
                # Still load them for combined analysis
                with open(output_path, "r") as f:
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