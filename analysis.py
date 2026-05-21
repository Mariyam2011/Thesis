"""
analysis.py
-----------
All analysis functions for the entropy thesis.

Computes:
  1. Accuracy
  2. AUROC for multiple entropy signals
  3. Baseline comparisons (random, output length)
  4. Per-segment entropy analysis (your core hypothesis)
  5. Length-confound control (partial correlation, residual AUROC)
  6. Statistical significance test
  7. Summary report

The key insight: we compare AUROC across signals to find which
entropy measure is most predictive of failure.
"""

import json
import random
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats


# ─────────────────────────────────────────────
# 1. Accuracy
# ─────────────────────────────────────────────

def compute_accuracy(results: list[dict]) -> dict:
    """Return accuracy and counts."""
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    return {
        "total":    total,
        "correct":  correct,
        "wrong":    total - correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
    }


# ─────────────────────────────────────────────
# 2. AUROC — core metric
# ─────────────────────────────────────────────

def compute_auroc_all_signals(results: list[dict]) -> dict:
    """
    Compute AUROC for every entropy signal.

    y_true = 1 if the model FAILED (wrong answer)
             0 if the model was correct

    Higher AUROC = entropy signal is better at predicting failure.
    Random baseline should be ~0.5.

    Returns a dict of signal_name → auroc_score.
    """
    y_true = [0 if r["correct"] else 1 for r in results]

    # Check we have both classes — AUROC is undefined otherwise
    if len(set(y_true)) < 2:
        print("[analysis] WARNING: only one class in y_true — AUROC undefined.")
        print(f"  Correct: {sum(1 for x in y_true if x==0)}, Wrong: {sum(1 for x in y_true if x==1)}")
        return {}

    signals = {
        # ── Primary entropy signals ──
        "entropy_mean":      [r["entropy_mean"]     for r in results],
        "entropy_max":       [r["entropy_max"]       for r in results],
        "entropy_last_10":   [r["entropy_last_10"]   for r in results],
        "entropy_first_10":  [r["entropy_first_10"]  for r in results],

        # ── Segment entropy signals (your core hypothesis) ──
        # These may be None if the model never produced that token type
        "entropy_arithmetic": [
            r["segment_entropy"].get("arithmetic") or r["entropy_mean"]
            for r in results
        ],
        "entropy_connector": [
            r["segment_entropy"].get("connector") or r["entropy_mean"]
            for r in results
        ],

        # ── Baselines ──
        "output_length": [
            len(r.get("generated_text", "")) for r in results
        ],
    }

    auroc_scores = {}
    for name, scores in signals.items():
        try:
            auroc = roc_auc_score(y_true, scores)
            auroc_scores[name] = round(auroc, 4)
        except Exception as e:
            auroc_scores[name] = None
            print(f"[analysis] AUROC failed for {name}: {e}")

    return auroc_scores


def compute_random_baseline_auroc(results: list[dict], n_trials: int = 200) -> dict:
    """
    Compute mean and std of AUROC for a random score predictor.
    Should be ≈ 0.5. Use this to confirm your entropy AUROC is meaningful.
    """
    y_true = [0 if r["correct"] else 1 for r in results]
    if len(set(y_true)) < 2:
        return {"mean": None, "std": None}

    scores = []
    for _ in range(n_trials):
        random_scores = [random.random() for _ in results]
        scores.append(roc_auc_score(y_true, random_scores))

    return {
        "mean": round(statistics.mean(scores), 4),
        "std":  round(statistics.stdev(scores), 4),
    }


# ─────────────────────────────────────────────
# 3. Per-segment entropy comparison
# ─────────────────────────────────────────────

def compare_segment_entropy_correct_vs_wrong(results: list[dict]) -> dict:
    """
    For each token type, compare mean entropy between correct and wrong answers.

    This directly tests: "Are arithmetic-step entropies higher in wrong answers?"

    Returns per-segment stats and a t-test p-value.
    """
    correct_results = [r for r in results if r["correct"]]
    wrong_results   = [r for r in results if not r["correct"]]

    segments = ["arithmetic", "connector", "other"]
    comparison = {}

    for seg in segments:
        correct_vals = [
            r["segment_entropy"].get(seg)
            for r in correct_results
            if r["segment_entropy"].get(seg) is not None
        ]
        wrong_vals = [
            r["segment_entropy"].get(seg)
            for r in wrong_results
            if r["segment_entropy"].get(seg) is not None
        ]

        if len(correct_vals) < 2 or len(wrong_vals) < 2:
            comparison[seg] = {"note": "insufficient data"}
            continue

        # Welch's t-test (does not assume equal variance)
        t_stat, p_value = stats.ttest_ind(wrong_vals, correct_vals, equal_var=False)

        comparison[seg] = {
            "correct_mean":  round(statistics.mean(correct_vals), 4),
            "correct_std":   round(statistics.stdev(correct_vals), 4),
            "wrong_mean":    round(statistics.mean(wrong_vals), 4),
            "wrong_std":     round(statistics.stdev(wrong_vals), 4),
            "delta":         round(statistics.mean(wrong_vals) - statistics.mean(correct_vals), 4),
            "t_statistic":   round(t_stat, 4),
            "p_value":       round(p_value, 6),
            "significant":   p_value < 0.05,
            "n_correct":     len(correct_vals),
            "n_wrong":       len(wrong_vals),
        }

    return comparison


# ─────────────────────────────────────────────
# 4. Entropy spike detection
# ─────────────────────────────────────────────

def find_entropy_spikes(entropies: list[float], threshold_multiplier: float = 1.5) -> list[int]:
    """
    Find positions in an entropy sequence where entropy exceeds
    mean + threshold_multiplier * std.

    Returns list of spike positions (indices).
    """
    if len(entropies) < 3:
        return []
    mean = statistics.mean(entropies)
    std  = statistics.stdev(entropies)
    threshold = mean + threshold_multiplier * std
    return [i for i, e in enumerate(entropies) if e > threshold]


def analyze_spike_positions(results: list[dict]) -> dict:
    """
    For correct vs wrong chains, analyze WHERE spikes tend to occur:
    - early (first 25% of chain)
    - middle (25–75%)
    - late  (last 25%)

    If wrong answers have more EARLY spikes, the model fails at
    problem setup. If more LATE spikes, it fails near the answer.
    """
    def position_bucket(idx, total):
        frac = idx / max(total - 1, 1)
        if frac < 0.25:   return "early"
        elif frac < 0.75: return "middle"
        else:             return "late"

    stats_dict = {
        "correct": {"early": 0, "middle": 0, "late": 0, "total_spikes": 0},
        "wrong":   {"early": 0, "middle": 0, "late": 0, "total_spikes": 0},
    }

    for r in results:
        entropies = r.get("entropies", [])
        if not entropies:
            continue
        key = "correct" if r["correct"] else "wrong"
        spikes = find_entropy_spikes(entropies)
        stats_dict[key]["total_spikes"] += len(spikes)
        for spike_idx in spikes:
            bucket = position_bucket(spike_idx, len(entropies))
            stats_dict[key][bucket] += 1

    return stats_dict


# ─────────────────────────────────────────────
# 5. Length confound — partial correlation & residual AUROC
# ─────────────────────────────────────────────

def _extract_length_confound_arrays(
    results: list[dict],
    *,
    require_arithmetic: bool = True,
    length_transform: str = "log1p",
) -> dict:
    """
    Per-problem arrays for length-confound analysis.
    y_fail = 1 if wrong, 0 if correct (same coding as AUROC).
    """
    rows = []
    for r in results:
        arith = r.get("segment_entropy", {}).get("arithmetic")
        if require_arithmetic and arith is None:
            continue
        if arith is None:
            arith = r["entropy_mean"]

        text = r.get("generated_text", "")
        length = len(text)
        if length_transform == "log1p":
            length_feat = float(np.log1p(length))
        elif length_transform == "raw":
            length_feat = float(length)
        else:
            raise ValueError(f"unknown length_transform: {length_transform}")

        rows.append((float(arith), 0.0 if r["correct"] else 1.0, length_feat, length))

    if not rows:
        raise ValueError("No rows after filtering (require_arithmetic=True?)")

    arith, y_fail, length_feat, raw_length = map(np.array, zip(*rows))
    return {
        "entropy_arithmetic": arith,
        "y_fail": y_fail,
        "length_feat": length_feat,
        "raw_length": raw_length,
        "n": len(rows),
        "length_transform": length_transform,
    }


def _residualize(y: np.ndarray, x_control: np.ndarray) -> np.ndarray:
    """OLS residuals of y ~ intercept + x_control."""
    X = np.column_stack([np.ones(len(x_control)), x_control])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """
    Pearson partial correlation r(x, y | z) via residualization.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    x_res = _residualize(x, z)
    y_res = _residualize(y, z)
    r, p = stats.pearsonr(x_res, y_res)
    return {"r_partial": round(float(r), 4), "p_value": float(p), "n": int(len(x))}


def length_adjusted_auroc(
    entropy: np.ndarray,
    y_fail: np.ndarray,
    length_feat: np.ndarray,
) -> float:
    """AUROC on arithmetic entropy residuals after regressing on length."""
    scores_res = _residualize(np.asarray(entropy, dtype=float), length_feat)
    if len(np.unique(y_fail)) < 2:
        return float("nan")
    return float(roc_auc_score(y_fail, scores_res))


def _bootstrap_ci(values: list[float], q: tuple[float, float] = (2.5, 97.5)) -> list[float | None]:
    if not values:
        return [None, None]
    lo, hi = np.percentile(values, q)
    return [round(float(lo), 4), round(float(hi), 4)]


def analyze_length_confound(
    results: list[dict],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    require_arithmetic: bool = True,
    length_transform: str = "log1p",
) -> dict:
    """
    Control for output length when testing arithmetic entropy vs failure.

    Returns partial correlation, raw/length/residual AUROCs, and bootstrap CIs.
    """
    base = _extract_length_confound_arrays(
        results,
        require_arithmetic=require_arithmetic,
        length_transform=length_transform,
    )
    x = base["entropy_arithmetic"]
    y = base["y_fail"]
    z = base["length_feat"]
    n = base["n"]

    pc_arith = partial_correlation(x, y, z)
    pc_length = partial_correlation(z, y, x)

    raw_auroc = float(roc_auc_score(y, x))
    len_auroc = float(roc_auc_score(y, z))
    adj_auroc = length_adjusted_auroc(x, y, z)

    rng = np.random.default_rng(seed)
    r_arith_boot, r_len_boot, raw_boot, adj_boot = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb, zb = x[idx], y[idx], z[idx]
        if len(np.unique(yb)) < 2:
            continue
        r_arith_boot.append(partial_correlation(xb, yb, zb)["r_partial"])
        r_len_boot.append(partial_correlation(zb, yb, xb)["r_partial"])
        raw_boot.append(roc_auc_score(yb, xb))
        adj_boot.append(length_adjusted_auroc(xb, yb, zb))

    return {
        "n_used": n,
        "n_total": len(results),
        "length_transform": length_transform,
        "partial_corr_arith_fail_given_length": {
            **pc_arith,
            "ci_95": _bootstrap_ci(r_arith_boot),
        },
        "partial_corr_length_fail_given_arith": {
            **pc_length,
            "ci_95": _bootstrap_ci(r_len_boot),
        },
        "auroc": {
            "entropy_arithmetic_raw": round(raw_auroc, 4),
            "output_length": round(len_auroc, 4),
            "entropy_arithmetic_length_adjusted": round(adj_auroc, 4),
            "raw_ci_95": _bootstrap_ci(raw_boot),
            "adjusted_ci_95": _bootstrap_ci(adj_boot),
        },
    }


# ─────────────────────────────────────────────
# 6. Full summary report
# ─────────────────────────────────────────────

def _make_json_safe(obj):
    import numpy as np

    # dict
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}

    # list / tuple
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]

    # numpy types (MAIN CULPRIT)
    elif isinstance(obj, np.bool_):
        return bool(obj)

    elif isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.floating):
        return float(obj)

    # fallback
    return obj

def generate_report(results: list[dict], output_path: str = "results/report.json") -> dict:
    """
    Generate and save a full analysis report.
    """
    print("[analysis] Generating report...")

    accuracy        = compute_accuracy(results)
    auroc_scores    = compute_auroc_all_signals(results)
    random_baseline = compute_random_baseline_auroc(results)
    segment_compare = compare_segment_entropy_correct_vs_wrong(results)
    spike_positions = analyze_spike_positions(results)
    length_confound = analyze_length_confound(results)

    report = {
        "n_problems":          len(results),
        "accuracy":            accuracy,
        "auroc_scores":        auroc_scores,
        "random_baseline_auroc": random_baseline,
        "segment_entropy_comparison": segment_compare,
        "spike_position_analysis":    spike_positions,
        "length_confound_analysis":   length_confound,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report = _make_json_safe(report)
    with open(output_path, "w") as f:
        json.dump(_make_json_safe(report), f, indent=2)

    _print_report(report)
    print(f"[analysis] Report saved -> {output_path}")
    return report


def _print_report(report: dict):
    """Pretty-print the key findings to console."""
    acc = report["accuracy"]
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  Problems:  {acc['total']}")
    print(f"  Correct:   {acc['correct']}  ({acc['accuracy']*100:.1f}%)")
    print(f"  Wrong:     {acc['wrong']}")

    print("\n-- AUROC Scores (failure prediction) --")
    print("  (higher = better predictor of failure | random ≈ 0.5)")
    rb = report["random_baseline_auroc"]
    print(f"  Random baseline: {rb['mean']} ± {rb['std']}")
    print()
    for signal, score in sorted(
        report["auroc_scores"].items(),
        key=lambda x: x[1] or 0,
        reverse=True
    ):
        bar = "#" * int((score or 0) * 20)
        print(f"  {signal:<25} {score:.4f}  {bar}")

    print("\n-- Segment Entropy: Correct vs Wrong --")
    for seg, data in report["segment_entropy_comparison"].items():
        if "note" in data:
            print(f"  {seg}: {data['note']}")
            continue
        sig = "***" if data["significant"] else ""
        print(
            f"  {seg:<12} "
            f"correct={data['correct_mean']:.3f}  "
            f"wrong={data['wrong_mean']:.3f}  "
            f"Δ={data['delta']:+.3f}  "
            f"p={data['p_value']:.4f} {sig}"
        )

    print("\n-- Entropy Spike Positions --")
    for outcome, pos in report["spike_position_analysis"].items():
        print(f"  {outcome}: early={pos['early']} mid={pos['middle']} late={pos['late']} total={pos['total_spikes']}")

    if "length_confound_analysis" in report:
        _print_length_confound(report["length_confound_analysis"])

    print("="*60 + "\n")


def _print_length_confound(lc: dict):
    print("\n-- Length Confound (arithmetic entropy vs failure) --")
    print(f"  n_used (with arithmetic segment): {lc['n_used']} / {lc['n_total']}")
    pa = lc["partial_corr_arith_fail_given_length"]
    pl = lc["partial_corr_length_fail_given_arith"]
    print(
        f"  Partial r(arith, fail | length): {pa['r_partial']:.4f}  "
        f"p={pa['p_value']:.2e}  CI={pa['ci_95']}"
    )
    print(
        f"  Partial r(length, fail | arith): {pl['r_partial']:.4f}  "
        f"p={pl['p_value']:.2e}  CI={pl['ci_95']}"
    )
    au = lc["auroc"]
    print(
        f"  AUROC raw arithmetic:     {au['entropy_arithmetic_raw']:.4f}  "
        f"CI={au['raw_ci_95']}"
    )
    print(f"  AUROC output length:      {au['output_length']:.4f}")
    print(
        f"  AUROC length-adjusted:    {au['entropy_arithmetic_length_adjusted']:.4f}  "
        f"CI={au['adjusted_ci_95']}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run thesis analysis on a results JSON file.")
    parser.add_argument(
        "results_path",
        nargs="?",
        default="results_ft/cot_gsm8k_full.json",
        help="Path to experiment results JSON (list of dicts)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Report output path (default: results_ft/length_confound_<stem>.json)",
    )
    parser.add_argument("--length-only", action="store_true", help="Only run length-confound analysis")
    args = parser.parse_args()

    results_path = Path(args.results_path)
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if args.length_only:
        out = args.output or str(
            results_path.parent / f"length_confound_{results_path.stem}.json"
        )
        report = analyze_length_confound(results)
        report = _make_json_safe(report)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[analysis] Length confound report saved -> {out}")
        _print_length_confound(report)
    else:
        out = args.output or str(results_path.parent / f"report_{results_path.stem}.json")
        generate_report(results, output_path=out)