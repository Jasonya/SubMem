"""Merge bigcodebench evaluator output into 07_metrics.json.

bigcodebench.evaluate writes two files next to the input JSONL:
  <sample>_eval_results.json  — per-task {status: pass|fail|timeout}
  <sample>_pass_at_k.json     — aggregate pass@k scores

We read both and inject pass_rate_true + breakdown into 07_metrics.json.

Usage:
    python scripts/bcb_update_metrics.py --submem-run-dir <dir>
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submem-run-dir", required=True)
    args = ap.parse_args()
    d = Path(args.submem_run_dir)

    metrics_path = d / "07_metrics.json"
    preds_path = d / "preds_bcb.jsonl"
    eval_results = d / "preds_bcb_eval_results.json"

    if not metrics_path.exists():
        raise SystemExit(f"ERR: {metrics_path} missing")
    if not eval_results.exists():
        # Common alt location: next to JSONL with _eval_results suffix
        alt = d / f"{preds_path.stem}_eval_results.json"
        if alt.exists():
            eval_results = alt
        else:
            raise SystemExit(f"ERR: no _eval_results.json found in {d}")

    eval_obj = json.load(open(eval_results))
    eval_map = eval_obj.get("eval", {})

    status_counter = Counter()
    task_status = {}
    for tid, runs in eval_map.items():
        if isinstance(runs, list) and runs:
            s = runs[0].get("status", "?")
        elif isinstance(runs, dict):
            s = runs.get("status", "?")
        else:
            s = "?"
        status_counter[s] += 1
        task_status[tid] = s

    n = sum(status_counter.values())
    n_pass = status_counter.get("pass", 0)
    pass_rate = n_pass / n if n else 0.0

    summary = {
        "pass_rate_true": pass_rate,
        "n_total": n,
        "n_pass": n_pass,
        "n_fail": status_counter.get("fail", 0),
        "n_timeout": status_counter.get("timeout", 0),
        "n_other": n - n_pass - status_counter.get("fail", 0) - status_counter.get("timeout", 0),
        "eval_results_path": str(eval_results),
    }

    # pass_at_k aggregate if present
    p_at_k_path = d / f"{preds_path.stem}_pass_at_k.json"
    if p_at_k_path.exists():
        try:
            summary["pass_at_k"] = json.load(open(p_at_k_path))
        except Exception:
            pass

    metrics = json.loads(metrics_path.read_text())
    metrics["bigcodebench_true"] = summary
    metrics["pass_rate_true"] = pass_rate
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nUpdated {metrics_path}")


if __name__ == "__main__":
    main()
