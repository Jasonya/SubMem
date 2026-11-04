"""Compute composite BCB-specific metrics from existing 07_metrics.json.

Reuses SubMem's E / Q families plus benchmark-specific pass_rate_true.
Adds under `bcb_composite`:
  P_hard_by_difficulty — pass@1 per {<15 min, 15m-1h, 1-4h, >4h}
  P_lib                — pass@1 per library (pandas/numpy/sklearn/...)
  D3_cumulative_auc    — area under cumulative-success curve vs task index
  TNS                  — token-normalized success = P_true * 1000 / E12_total
  LWS                  — latency-weighted success = P_true * exp(-E2_p99/10s)

Usage:
  python scripts/bcb_compute_composite_metrics.py \
    [--runs-glob '$SUBMEM_RESULTS/*_bcb_*/*/07_metrics.json']
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path

import pandas as pd

# Repo root: $SUBMEM_ROOT, else derived from this file's location (scripts/..).
SUBMEM_ROOT = Path(os.environ.get("SUBMEM_ROOT", Path(__file__).resolve().parents[1]))
SUBMEM_RESULTS = Path(os.environ.get("SUBMEM_RESULTS", SUBMEM_ROOT / "results"))
# Hugging Face cache root: $HF_HOME, else the HF default ~/.cache/huggingface.
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _bcb_dataset():
    for pq in [
        HF_HOME / "hub/datasets--bigcode--bigcodebench/snapshots/b74c0d0bf70d2c0bc459be537895cca163007f1a/data/train-00000-of-00001.parquet",
    ]:
        if Path(pq).exists():
            return pd.read_parquet(pq)
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id="bigcode/bigcodebench", repo_type="dataset",
                          allow_patterns=["*.parquet"])
    return pd.read_parquet(list(Path(p).rglob("*.parquet"))[0])


def _libs_of(task) -> list[str]:
    libs = task.get("libs")
    if libs is None:
        return []
    if isinstance(libs, str):
        try:
            return json.loads(libs)
        except Exception:
            return libs.split(",")
    return list(libs)


def compute_composite(metrics_path: Path, bcb_map: dict) -> dict:
    m = json.loads(metrics_path.read_text())
    run_dir = metrics_path.parent
    eval_path = run_dir / "preds_bcb_eval_results.json"
    preds_path = run_dir / "05_predictions.json"
    if not eval_path.exists() or not preds_path.exists():
        return {}

    eval_obj = json.loads(eval_path.read_text())
    eval_map = eval_obj.get("eval", {})
    preds = json.loads(preds_path.read_text())

    # Per-task pass/fail in the order the driver produced them
    ordered_pass = []
    per_difficulty = {}
    per_lib = {}
    for p in preds:
        tid = p["task_id"]
        status_entry = eval_map.get(tid)
        if isinstance(status_entry, list) and status_entry:
            status = status_entry[0].get("status", "?")
        elif isinstance(status_entry, dict):
            status = status_entry.get("status", "?")
        else:
            status = "?"
        ok = (status == "pass")
        ordered_pass.append(ok)

        task = bcb_map.get(tid)
        if task is None:
            continue
        diff = str(task.get("difficulty", "unknown"))
        per_difficulty.setdefault(diff, []).append(ok)
        for lib in _libs_of(task):
            per_lib.setdefault(lib, []).append(ok)

    # D3' cumulative success AUC (normalised to [0,1])
    n = len(ordered_pass)
    if n:
        csum = 0
        auc = 0.0
        for t, ok in enumerate(ordered_pass, start=1):
            csum += int(ok)
            auc += csum / t
        d3_auc = auc / n
    else:
        d3_auc = None

    eff = m.get("efficiency") or {}
    p_true = m.get("pass_rate_true") or 0.0
    e12 = eff.get("E12_total_tokens") or 0
    e2p99 = (eff.get("E2_inference_ms_p90")  # E2_p99 not stored; use p90 as proxy
             or 0)

    tns = (p_true * 1000.0 / e12) if e12 else None
    lws = p_true * math.exp(-e2p99 / 10000.0) if e2p99 else p_true

    composite = {
        "P_true": round(p_true, 4),
        "P_by_difficulty": {k: round(sum(v)/len(v), 4) for k, v in per_difficulty.items()},
        "N_by_difficulty": {k: len(v) for k, v in per_difficulty.items()},
        "P_by_lib_top10": dict(sorted(
            [(k, round(sum(v)/len(v), 4)) for k, v in per_lib.items() if len(v) >= 5],
            key=lambda kv: -kv[1])[:10]),
        "D3_cumulative_auc": round(d3_auc, 4) if d3_auc is not None else None,
        "TNS_pass_per_1k_tokens": round(tns, 4) if tns is not None else None,
        "LWS_latency_weighted_pass": round(lws, 4) if lws is not None else None,
        "_inputs": {
            "E12_total_tokens": e12,
            "E2_p90_ms": e2p99,
            "n_tasks": n,
        },
    }
    return composite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob",
                    default=str(SUBMEM_RESULTS / "*_bcb_*/*/07_metrics.json"))
    args = ap.parse_args()

    bcb = _bcb_dataset()
    bcb_map = {r["task_id"]: r for _, r in bcb.iterrows()}

    n_updated = 0
    for mp in sorted(glob.glob(args.runs_glob)):
        mpath = Path(mp)
        try:
            composite = compute_composite(mpath, bcb_map)
        except Exception as e:
            print(f"  {mpath.parent.name}: ERR {type(e).__name__}: {e}")
            continue
        if not composite:
            continue
        m = json.loads(mpath.read_text())
        m["bcb_composite"] = composite
        mpath.write_text(json.dumps(m, indent=2))
        n_updated += 1

    print(f"Updated {n_updated} metrics files with `bcb_composite`.")


if __name__ == "__main__":
    main()
