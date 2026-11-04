"""Analyze ALFWorld rollout runs — 7 ALFWorld-specific metrics.

Reads per_task.json from one or more rollout run dirs + substrate IDs, computes
side-by-side comparison table. Optional bank-size measurement if --bank-dir
supplied for each entry.

Metrics:
  1. TSR            Task Success Rate            sum(succ)/N
  2. By-type SR     per task-type breakdown      dict
  3. Avg steps      success-only avg step count  mean(steps) where success
  4. Hit-max rate   fraction that hit max_steps  sum(steps==max_steps) / N
  5. Delta-SR       vs a --baseline run dir      TSR(this) - TSR(baseline)
  6. Tokens/task    eval prompt+generation       mean(len(raw))+len(prompt)
                    (prompt length inferred from trajectory meta if present)
  7. Bank bytes     pkl file size                os.path.getsize(bank_Cx.pkl)

Usage:
  python analyze_alfworld_run.py \
      --run name=M10 run_dir=results/20260419_195306_alfworld_Qwen3-32B-AWQ memory_id=M10 \
      --run name=M7_k33 run_dir=results/20260420_044213... memory_id=M7 bank_dir=results/alfworld_banks_c14 \
      --run name=M7_k11 run_dir=results/... memory_id=M7 bank_dir=results/alfworld_banks_c14 \
      --baseline M10
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

ALF_TASK_KEYS = ("look", "two", "clean", "cool", "heat", "put")


def classify_task_type(goal: str) -> str:
    g = goal.lower()
    for k in ALF_TASK_KEYS:
        if k in g:
            return k
    return "other"


def parse_run_spec(spec: List[str]) -> Dict[str, str]:
    """Parse CLI args of the form `name=X run_dir=Y memory_id=Z` into a dict."""
    out = {}
    for s in spec:
        if "=" not in s:
            raise SystemExit(f"bad run spec fragment: {s!r}")
        k, v = s.split("=", 1)
        out[k] = v
    for required in ("name", "run_dir", "memory_id"):
        if required not in out:
            raise SystemExit(f"run spec missing {required}: {spec}")
    return out


def load_tasks(run_dir: str, memory_id: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, memory_id, "per_task.json")
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")
    return json.load(open(path))


def compute_metrics(tasks: List[Dict[str, Any]], bank_bytes: Optional[int],
                    max_steps: int = 50) -> Dict[str, Any]:
    n = len(tasks)
    if n == 0:
        return {"n": 0}

    # 1. TSR
    n_succ = sum(int(t.get("success", False)) for t in tasks)
    tsr = n_succ / n

    # 2. By-type
    by_type: Dict[str, List[int]] = defaultdict(list)
    for t in tasks:
        tt = classify_task_type(t.get("goal", ""))
        by_type[tt].append(int(t.get("success", False)))
    by_type_stats = {
        tt: {"n": len(v), "sr": round(sum(v) / len(v), 4)}
        for tt, v in sorted(by_type.items())
    }

    # 3. Avg steps (success-only)
    succ_steps = [t.get("steps", 0) for t in tasks if t.get("success")]
    fail_steps = [t.get("steps", 0) for t in tasks if not t.get("success")]
    avg_steps_succ = statistics.mean(succ_steps) if succ_steps else None
    avg_steps_fail = statistics.mean(fail_steps) if fail_steps else None

    # 4. Hit-max rate
    hit_max = sum(1 for t in tasks if t.get("steps", 0) >= max_steps)
    hit_max_rate = hit_max / n

    # 6. Tokens per task (approx): avg(sum(len(raw)) + sum(len(obs)) per task)
    # We use character counts as proxy when token counts aren't stored.
    tokens_per_task = []
    for t in tasks:
        traj = t.get("trajectory", [])
        # Sum of raw completion chars + obs chars as proxy for LLM-seen tokens
        chars = sum(len(h.get("raw", "")) + len(h.get("obs", "")) for h in traj)
        tokens_per_task.append(chars // 4)  # ~4 chars per token
    avg_tokens_per_task = statistics.mean(tokens_per_task) if tokens_per_task else None

    # Repetitive action rate (bonus, not in top-7 but cheap)
    repetitive_count = 0
    for t in tasks:
        actions = [h.get("action", "") for h in t.get("trajectory", [])]
        for i in range(len(actions) - 2):
            if actions[i] == actions[i+1] == actions[i+2]:
                repetitive_count += 1
                break
    repetitive_rate = repetitive_count / n

    return {
        "n": n,
        "n_succ": n_succ,
        "tsr": tsr,
        "by_type": by_type_stats,
        "avg_steps_succ": avg_steps_succ,
        "avg_steps_fail": avg_steps_fail,
        "hit_max_rate": hit_max_rate,
        "avg_tokens_per_task": avg_tokens_per_task,
        "repetitive_rate": repetitive_rate,
        "bank_bytes": bank_bytes,
    }


def find_bank_size(bank_dir: Optional[str], memory_id: str) -> Optional[int]:
    if not bank_dir:
        return None
    path = os.path.join(bank_dir, f"bank_{memory_id}.pkl")
    if not os.path.exists(path):
        return None
    return os.path.getsize(path)


def fmt(v, pct: bool = False, decimals: int = 1) -> str:
    if v is None:
        return "-"
    if pct:
        return f"{100 * v:.{decimals}f}%"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def print_table(runs: List[Dict[str, Any]], baseline_name: Optional[str]) -> None:
    # Primary table
    headers = ["name", "n", "TSR", "Δ-TSR", "steps_s", "steps_f", "hit_max",
               "tok/task", "rep_act", "bank_MB"]
    rows = []
    baseline_tsr = None
    if baseline_name:
        for r in runs:
            if r["spec"]["name"] == baseline_name:
                baseline_tsr = r["metrics"].get("tsr")
                break
        if baseline_tsr is None:
            print(f"WARN: baseline {baseline_name!r} not found among runs")

    for r in runs:
        m = r["metrics"]
        row = [
            r["spec"]["name"],
            fmt(m.get("n")),
            fmt(m.get("tsr"), pct=True, decimals=2),
            fmt(m["tsr"] - baseline_tsr, pct=True, decimals=2)
                if baseline_tsr is not None and m.get("tsr") is not None else "-",
            fmt(m.get("avg_steps_succ"), decimals=1),
            fmt(m.get("avg_steps_fail"), decimals=1),
            fmt(m.get("hit_max_rate"), pct=True),
            fmt(m.get("avg_tokens_per_task"), decimals=0),
            fmt(m.get("repetitive_rate"), pct=True),
            fmt(m["bank_bytes"] / 1024 / 1024, decimals=2)
                if m.get("bank_bytes") else "-",
        ]
        rows.append(row)

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = " | "
    print("\n=== Primary Metrics ===")
    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep.join("-" * w for w in widths))
    for row in rows:
        print(sep.join(c.ljust(w) for c, w in zip(row, widths)))

    # By-type table
    print("\n=== By Task Type (success rate) ===")
    all_types = set()
    for r in runs:
        all_types.update(r["metrics"].get("by_type", {}).keys())
    all_types = sorted(all_types)

    # Collect n for each type from first run that has it (they should match)
    type_n = {}
    for tt in all_types:
        for r in runs:
            bt = r["metrics"].get("by_type", {})
            if tt in bt:
                type_n[tt] = bt[tt]["n"]
                break

    type_headers = ["type", "n"] + [r["spec"]["name"] for r in runs]
    type_rows = []
    for tt in all_types:
        row = [tt, str(type_n.get(tt, "?"))]
        for r in runs:
            bt = r["metrics"].get("by_type", {}).get(tt)
            row.append(fmt(bt["sr"] if bt else None, pct=True, decimals=1))
        type_rows.append(row)
    type_widths = [max(len(h), *(len(row[i]) for row in type_rows))
                   for i, h in enumerate(type_headers)]
    print(sep.join(h.ljust(w) for h, w in zip(type_headers, type_widths)))
    print(sep.join("-" * w for w in type_widths))
    for row in type_rows:
        print(sep.join(c.ljust(w) for c, w in zip(row, type_widths)))

    # Δ-per-type vs baseline
    if baseline_name and baseline_tsr is not None:
        baseline_bt = next(
            (r["metrics"]["by_type"] for r in runs if r["spec"]["name"] == baseline_name), {}
        )
        print(f"\n=== Δ By-Type vs {baseline_name} ===")
        dtype_headers = ["type"] + [r["spec"]["name"] for r in runs
                                    if r["spec"]["name"] != baseline_name]
        dtype_rows = []
        for tt in all_types:
            base_sr = baseline_bt.get(tt, {}).get("sr")
            row = [tt]
            for r in runs:
                if r["spec"]["name"] == baseline_name:
                    continue
                sr = r["metrics"].get("by_type", {}).get(tt, {}).get("sr")
                if base_sr is None or sr is None:
                    row.append("-")
                else:
                    row.append(fmt(sr - base_sr, pct=True, decimals=1))
            dtype_rows.append(row)
        dt_widths = [max(len(h), *(len(row[i]) for row in dtype_rows))
                     for i, h in enumerate(dtype_headers)]
        print(sep.join(h.ljust(w) for h, w in zip(dtype_headers, dt_widths)))
        print(sep.join("-" * w for w in dt_widths))
        for row in dtype_rows:
            print(sep.join(c.ljust(w) for c, w in zip(row, dt_widths)))


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--run", action="append", nargs="+", required=True,
                    help="Repeat: --run name=X run_dir=Y memory_id=Z [bank_dir=W]")
    ap.add_argument("--baseline", default=None,
                    help="Name of the run to use as baseline for Δ-TSR")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--json-out", default=None,
                    help="Also dump full metrics JSON here")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        s = parse_run_spec(spec)
        tasks = load_tasks(s["run_dir"], s["memory_id"])
        bank_bytes = find_bank_size(s.get("bank_dir"), s["memory_id"])
        m = compute_metrics(tasks, bank_bytes, max_steps=args.max_steps)
        runs.append({"spec": s, "metrics": m})
        print(f"[{s['name']}] loaded {m['n']} tasks from {s['run_dir']}/{s['memory_id']}")

    print_table(runs, baseline_name=args.baseline)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(runs, f, indent=2, default=str)
        print(f"\nFull metrics written to {args.json_out}")


if __name__ == "__main__":
    main()
