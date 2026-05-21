"""
run_lora_rank_experiment.py
-----------------------------
Train and/or evaluate LoRA adapters at multiple ranks to compare r=16 vs r=32 vs r=64.

Typical workflow (after you already trained r=16 at lora_outputs/qwen3_gsm8k/adapter):

  # Train r=32 and r=64 only (alpha = 2*r by default)
  python run_lora_rank_experiment.py --ranks 32 64 --train-only

  # Evaluate all available adapters on GSM8K test (full run)
  python run_lora_rank_experiment.py --ranks 16 32 64 --eval-only

  # Train + evaluate in one go
  python run_lora_rank_experiment.py --ranks 32 64

  # Smoke test: 50 problems per rank
  python run_lora_rank_experiment.py --ranks 32 64 --limit 50
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from finetune_lora import output_dir_for_rank, train_lora


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_BASE = Path("lora_outputs/qwen3_gsm8k")
DEFAULT_RESULTS_ROOT = Path("results_lora_rank")
DEFAULT_BASELINE_ADAPTER = Path("lora_outputs/qwen3_gsm8k/adapter")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA rank ablation: train and evaluate r=16/32/64 adapters."
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[32, 64],
        help="LoRA ranks to train/evaluate (default: 32 64).",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also evaluate existing r=16 adapter at lora_outputs/qwen3_gsm8k/adapter.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-base",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Base output dir; per-rank dirs become <base>_r<rank>/adapter.",
    )
    parser.add_argument(
        "--baseline-adapter",
        type=str,
        default=str(DEFAULT_BASELINE_ADAPTER),
        help="Path to existing r=16 adapter when using --include-baseline.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root folder for per-rank inference results.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="lora_outputs/lora_rank_experiment_summary.json",
        help="Where to write the cross-rank comparison JSON.",
    )

    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip train/eval if outputs exist.")

    # Forwarded to finetune_lora / main
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-alpha-scale", type=float, default=2.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None, help="Max GSM8K test problems for eval.")
    parser.add_argument("--strategies", nargs="+", default=["cot"])
    parser.add_argument("--force-rerun", action="store_true", help="Overwrite existing eval result files.")
    return parser.parse_args()


def adapter_dir_for_rank(output_base: Path, rank: int) -> Path:
    return output_dir_for_rank(output_base, rank, multi_rank=True) / "adapter"


def build_finetune_namespace(exp_args) -> Namespace:
    """Build an argparse.Namespace compatible with finetune_lora.train_lora."""
    return Namespace(
        model=exp_args.model,
        output_dir=str(exp_args.output_base),
        max_length=exp_args.max_length,
        seed=exp_args.seed,
        epochs=exp_args.epochs,
        lr=exp_args.lr,
        batch_size=exp_args.batch_size,
        grad_accum=exp_args.grad_accum,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=20,
        save_steps=200,
        eval_steps=200,
        lora_r=16,
        lora_ranks=None,
        lora_alpha=None,
        lora_alpha_scale=exp_args.lora_alpha_scale,
        lora_dropout=exp_args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        save_merged=False,
    )


def ranks_for_training(exp_args) -> list[int]:
    ranks = sorted(set(exp_args.ranks))
    if exp_args.include_baseline and 16 in ranks:
        ranks = [r for r in ranks if r != 16]
    return ranks


def collect_rank_jobs(exp_args) -> list[dict]:
    """Return list of {rank, adapter_path, results_dir, label}."""
    output_base = Path(exp_args.output_base)
    results_root = Path(exp_args.results_root)
    jobs = []

    if exp_args.include_baseline:
        baseline = Path(exp_args.baseline_adapter)
        if baseline.exists():
            jobs.append(
                {
                    "rank": 16,
                    "adapter_path": str(baseline),
                    "results_dir": str(results_root / "r16"),
                    "label": "baseline_r16",
                }
            )
        else:
            print(f"[rank-exp] WARNING: baseline adapter not found: {baseline}")

    for rank in sorted(set(exp_args.ranks)):
        adapter = adapter_dir_for_rank(output_base, rank)
        jobs.append(
            {
                "rank": rank,
                "adapter_path": str(adapter),
                "results_dir": str(results_root / f"r{rank}"),
                "label": f"r{rank}",
            }
        )

    # Deduplicate by rank (baseline may overlap if 16 in --ranks)
    seen = set()
    unique = []
    for job in jobs:
        if job["rank"] in seen:
            continue
        seen.add(job["rank"])
        unique.append(job)
    return unique


def train_ranks(exp_args) -> list[dict]:
    ft_args = build_finetune_namespace(exp_args)
    output_base = Path(exp_args.output_base)
    train_summaries = []

    for rank in ranks_for_training(exp_args):
        out_dir = output_dir_for_rank(output_base, rank, multi_rank=True)
        adapter_dir = out_dir / "adapter"
        if exp_args.skip_existing and adapter_dir.exists():
            print(f"[rank-exp] Skip training r={rank} (adapter exists): {adapter_dir}")
            continue

        print(f"\n[rank-exp] === Training LoRA r={rank} ===")
        summary = train_lora(ft_args, lora_r=rank, output_dir=out_dir)
        train_summaries.append(summary)

    return train_summaries


def run_eval_for_job(exp_args, job: dict) -> dict | None:
    adapter = Path(job["adapter_path"])
    if not adapter.exists():
        print(f"[rank-exp] Skip eval r={job['rank']}: missing adapter {adapter}")
        return None

    results_dir = Path(job["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    full_path = results_dir / "cot_gsm8k_full.json"

    if exp_args.skip_existing and full_path.exists() and not exp_args.force_rerun:
        print(f"[rank-exp] Skip eval r={job['rank']} (results exist): {full_path}")
    else:
        cmd = [
            sys.executable,
            "main.py",
            "--model",
            exp_args.model,
            "--adapter-path",
            str(adapter),
            "--strategies",
            *exp_args.strategies,
            "--datasets",
            "gsm8k",
            "--results-dir",
            str(results_dir),
        ]
        if exp_args.limit is not None:
            cmd.extend(["--limit", str(exp_args.limit)])
        if exp_args.force_rerun:
            cmd.append("--force-rerun")

        print(f"\n[rank-exp] === Evaluating r={job['rank']} ===")
        print("[rank-exp] Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    report_path = results_dir / "report_cot_gsm8k.json"
    if not report_path.exists():
        print(f"[rank-exp] WARNING: no report at {report_path}")
        return {"rank": job["rank"], "label": job["label"], "adapter_path": str(adapter), "error": "no report"}

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    return {
        "rank": job["rank"],
        "label": job["label"],
        "adapter_path": str(adapter),
        "results_dir": str(results_dir),
        "accuracy": report.get("accuracy", {}).get("accuracy"),
        "auroc_entropy_arithmetic": report.get("auroc_scores", {}).get("entropy_arithmetic"),
        "auroc_output_length": report.get("auroc_scores", {}).get("output_length"),
        "auroc_entropy_mean": report.get("auroc_scores", {}).get("entropy_mean"),
        "length_confound": report.get("length_confound_analysis"),
    }


def save_summary(exp_args, payload: dict) -> Path:
    path = Path(exp_args.summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[rank-exp] Summary saved -> {path.resolve()}")
    return path


def print_comparison(eval_rows: list[dict]):
    rows = [r for r in eval_rows if r and "error" not in r]
    if not rows:
        print("[rank-exp] No evaluation rows to compare.")
        return

    print("\n" + "=" * 72)
    print("  LoRA RANK EXPERIMENT COMPARISON")
    print("=" * 72)
    print(f"  {'rank':<6} {'accuracy':>10} {'arith_AUROC':>12} {'length_AUROC':>13}")
    print("  " + "-" * 68)
    for row in sorted(rows, key=lambda x: x["rank"]):
        acc = row.get("accuracy")
        arith = row.get("auroc_entropy_arithmetic")
        length = row.get("auroc_output_length")
        acc_s = f"{acc*100:.1f}%" if acc is not None else "n/a"
        arith_s = f"{arith:.4f}" if arith is not None else "n/a"
        length_s = f"{length:.4f}" if length is not None else "n/a"
        print(f"  r={row['rank']:<4} {acc_s:>10} {arith_s:>12} {length_s:>13}")
    print("=" * 72 + "\n")


def main():
    exp_args = parse_args()
    if exp_args.train_only and exp_args.eval_only:
        print("[rank-exp] ERROR: use only one of --train-only or --eval-only")
        sys.exit(1)

    do_train = not exp_args.eval_only
    do_eval = not exp_args.train_only

    payload = {
        "model": exp_args.model,
        "ranks_requested": exp_args.ranks,
        "include_baseline": exp_args.include_baseline,
        "output_base": exp_args.output_base,
        "lora_alpha_scale": exp_args.lora_alpha_scale,
        "training": [],
        "evaluation": [],
    }

    if do_train:
        payload["training"] = train_ranks(exp_args)

    if do_eval:
        jobs = collect_rank_jobs(exp_args)
        for job in jobs:
            row = run_eval_for_job(exp_args, job)
            if row:
                payload["evaluation"].append(row)
        print_comparison(payload["evaluation"])

    save_summary(exp_args, payload)


if __name__ == "__main__":
    main()
