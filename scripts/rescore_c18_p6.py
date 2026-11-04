"""Offline P6 / P7 rescoring for M5 / formerly C18 (LightRAG).

Re-runs the dia_id attribution layered strategy against the saved
LightRAG answers in 03_retrievals.json — no model calls needed.
Also reads the corresponding sample_id's writes to recover the chunk
list and dia_id mapping.

Usage:
    python scripts/rescore_c18_p6.py <result_dir>

Or pass a glob:
    python scripts/rescore_c18_p6.py --glob 'results/*locomo*/[CM]*'
"""
from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob


_TS_SPEAKER = re.compile(
    r"\[?\s*([0-9]{1,2}[:\s][0-9 apmAPM:.,/-]+[0-9])\s*\]?\s*"
    r"([A-Z][A-Za-z' ]{1,30}?)\s*[:：]"
)


def _norm_tokens(s: str) -> set:
    return set(re.findall(r"[A-Za-z0-9]+", s.lower()))


def _attribute(answer: str, chunks: list[str], dia_ids: list[str],
               top_k: int = 10) -> list[str]:
    """Return up to top_k dia_ids attributed to this answer."""
    out: list[str] = []
    seen: set[str] = set()

    if not answer or not chunks:
        return out

    # Strategy 1: verbatim substring
    for i, chunk in enumerate(chunks):
        if chunk and chunk[:80] in answer:
            did = dia_ids[i] if i < len(dia_ids) else ""
            if did and did not in seen:
                out.append(did); seen.add(did)
                if len(out) >= top_k:
                    return out

    # Strategy 2/3: parse References block
    ref_block = ""
    if "### References" in answer:
        ref_block = answer.split("### References", 1)[1]
    ref_lines = [
        ln.strip().lstrip("-*•").strip()
        for ln in ref_block.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    chunk_sigs = []
    for chunk in chunks:
        if not chunk:
            chunk_sigs.append(None); continue
        m = _TS_SPEAKER.search(chunk[:200])
        sig = (
            (m.group(1).strip().lower() if m else None,
             m.group(2).strip().lower() if m else None,
             _norm_tokens(chunk))
        )
        chunk_sigs.append(sig)

    for line in ref_lines:
        if len(out) >= top_k:
            break
        line_tokens = _norm_tokens(line)
        if not line_tokens:
            continue
        m = _TS_SPEAKER.search(line)
        line_ts = m.group(1).strip().lower() if m else None
        line_speaker = m.group(2).strip().lower() if m else None

        best_i, best_sim = -1, 0.0
        for i, sig in enumerate(chunk_sigs):
            if sig is None: continue
            ts, sp, ctokens = sig
            if line_ts and line_speaker and ts and sp:
                if line_speaker == sp and (line_ts in ts or ts in line_ts):
                    best_i, best_sim = i, 1.0
                    break
            inter = len(line_tokens & ctokens)
            union = len(line_tokens | ctokens)
            sim = inter / union if union else 0.0
            if sim > best_sim:
                best_i, best_sim = i, sim

        if best_i >= 0 and best_sim >= 0.5:
            did = dia_ids[best_i] if best_i < len(dia_ids) else ""
            if did and did not in seen:
                out.append(did); seen.add(did)

    return out


def _recompute_dir(rd: str, top_k: int = 10) -> dict | None:
    ret_p = os.path.join(rd, "03_retrievals.json")
    if not os.path.isfile(ret_p):
        return None
    rets = json.load(open(ret_p))

    # Build (sample_id) -> (chunks[], dia_ids[]) from per-sample writes.
    # Schema: each `02_writes_sample{i}.json` is a dict
    #   {"sample_id": ..., "entries": [{"text", "metadata": {"dia_id"}}, ...]}
    sid_to_chunks: dict[str, list[str]] = {}
    sid_to_dias: dict[str, list[str]] = {}

    sample_files = sorted(
        f for f in os.listdir(rd)
        if f.startswith("02_writes_sample") and f.endswith(".json")
    )
    for sf in sample_files:
        try:
            ws = json.load(open(os.path.join(rd, sf)))
        except Exception:
            continue
        sid = ws.get("sample_id", "")
        chunks, dias = [], []
        for e in ws.get("entries", []):
            chunks.append(e.get("text", ""))
            dias.append((e.get("metadata") or {}).get("dia_id", ""))
        sid_to_chunks[sid] = chunks
        sid_to_dias[sid] = dias

    # Fallback: 02_writes_all.json (older list-of-writes format)
    if not sid_to_chunks:
        wp = os.path.join(rd, "02_writes_all.json")
        if os.path.isfile(wp):
            ws = json.load(open(wp))
            if isinstance(ws, list):
                for w in ws:
                    sid = w.get("sample_id", "")
                    sid_to_chunks.setdefault(sid, []).append(w.get("text", ""))
                    sid_to_dias.setdefault(sid, []).append(
                        (w.get("metadata") or {}).get("dia_id", "")
                    )

    n_with_gold = 0
    recall_sum = 0.0
    prec_sum = 0.0
    hits = 0
    for r in rets:
        gold = set(r.get("gold_evidence") or [])
        if not gold:
            continue
        n_with_gold += 1
        sid = r.get("sample_id", "")
        chunks = sid_to_chunks.get(sid, [])
        dias = sid_to_dias.get(sid, [])
        # take first retrieved entry's content (LightRAG returns 1 synthesised entry)
        retrieved = r.get("retrieved", [])
        answer = retrieved[0].get("content", "") if retrieved else ""
        new_dias = _attribute(answer, chunks, dias, top_k=top_k)
        new_set = set(new_dias)
        inter = gold & new_set
        if inter:
            hits += 1
        recall_sum += len(inter) / max(len(gold), 1)
        prec_sum += len(inter) / max(len(new_set), 1) if new_set else 0.0

    if n_with_gold == 0:
        return None
    return {
        "n_with_gold": n_with_gold,
        "P6_recall_at_k": recall_sum / n_with_gold,
        "P7_precision_at_k": prec_sum / n_with_gold,
        "hit_rate": hits / n_with_gold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="single result/<run>/C18 dir")
    ap.add_argument("--glob", help="glob pattern, e.g. 'results/*locomo*/[CM]*'")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    targets = []
    if args.glob:
        targets = sorted(glob(args.glob))
    elif args.path:
        targets = [args.path]
    if not targets:
        print("no targets")
        return

    print(f"{'P6':>7} {'P7':>7} {'hit%':>6} {'n':>4}  dir")
    for d in targets:
        out = _recompute_dir(d, top_k=args.top_k)
        if out is None:
            print(f"      -       -      -    -  {d}  (no retrievals/writes)")
            continue
        print(f"  {out['P6_recall_at_k']:.4f}  {out['P7_precision_at_k']:.4f}  "
              f"{out['hit_rate']*100:5.1f}  {out['n_with_gold']:>4}  {d}")


if __name__ == "__main__":
    main()
