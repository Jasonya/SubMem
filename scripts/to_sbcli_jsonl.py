"""Convert our driver's 05_predictions.json into sb-cli's JSONL format.

Each line: {"instance_id", "model_name_or_path", "model_patch"}.

The driver stores the raw LLM response in ``prediction``. LLMs often wrap
the patch in a markdown ``` ```diff ... ``` ``` fence with surrounding prose.
We extract the diff block; if no fence is present, we heuristically slice
from the first ``diff --git`` line.

Usage:
    python scripts/to_sbcli_jsonl.py \
        --predictions-json <run_dir>/<CFG>/05_predictions.json \
        --model-name submem-<CFG> \
        --output <run_dir>/<CFG>/preds_sbcli.jsonl
"""
import argparse
import json
import re
import sys


_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)


def extract_patch(text: str) -> str:
    """Pull a unified diff out of an LLM response; return '' if none found."""
    if not text:
        return ""
    # 1) fenced code block — prefer the one that contains 'diff --git'
    for m in _FENCE_RE.finditer(text):
        block = m.group(1)
        if "diff --git" in block or block.lstrip().startswith(("---", "+++", "@@")):
            return block.strip() + "\n"
    # 2) fallback: first 'diff --git' line onward, stopping at a closing fence
    i = text.find("diff --git")
    if i >= 0:
        rest = text[i:]
        j = rest.find("\n```")
        return (rest[:j] if j > 0 else rest).strip() + "\n"
    # 3) bare hunk starting with --- a/
    i = text.find("\n--- a/")
    if i >= 0:
        rest = text[i + 1:]
        j = rest.find("\n```")
        return (rest[:j] if j > 0 else rest).strip() + "\n"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-json", required=True)
    ap.add_argument("--model-name", required=True,
                    help="model_name_or_path field (e.g. submem-C1-gpt4omini)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.predictions_json) as f:
        preds = json.load(f)

    n_empty = 0
    with open(args.output, "w") as f:
        for p in preds:
            patch = extract_patch(p.get("prediction", ""))
            if not patch:
                n_empty += 1
            f.write(json.dumps({
                "instance_id": p["instance_id"],
                "model_name_or_path": args.model_name,
                "model_patch": patch,
            }) + "\n")

    print(f"Wrote {len(preds)} predictions → {args.output}", file=sys.stderr)
    print(f"  empty (no diff extracted): {n_empty}", file=sys.stderr)


if __name__ == "__main__":
    main()
