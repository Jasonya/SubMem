#!/bin/bash
# Apply SubMem's M8 token-instrumentation patch to the vendored
# ViktorAxelsen/MemSkill checkout.
#
# WHY THIS EXISTS
#   third_party/ is .gitignore'd, so the vendored MemSkill repo is not
#   under version control. The M8 substrate
#   (memory/c20_memskill_faithful.py) runs that repo as an opaque
#   subprocess; on its own it cannot see the pipeline's LLM token usage.
#   This patch adds a per-component token meter so M8 can recover token
#   counts — split by upstream component — via a sidecar JSON, and map
#   them to SubMem's write/mgmt E-metric buckets.
#
#   Components metered: controller, executor (split write/mgmt by action
#   count), designer. The judge is deliberately NOT metered — it is eval
#   cost, not a memory operation.
#
#   Run this once after cloning / rebuilding third_party/MemSkill.
#
# USAGE
#   bash scripts/apply_m8_token_patch.sh
#
# IDEMPOTENT: re-running detects the marker and skips already-patched files.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$REPO_ROOT/third_party/MemSkill"

if [[ ! -d "$VENDOR" ]]; then
    echo "ERR: vendored MemSkill not found at $VENDOR" >&2
    echo "     Run scripts/vendor_memskill.sh first." >&2
    exit 1
fi

# ── 1. Write the token-meter module (idempotent: overwrite is fine) ──
cat > "$VENDOR/src/_submem_token_meter.py" <<'METER_EOF'
"""SubMem instrumentation patch — process-wide LLM token meter.

NOT part of upstream ViktorAxelsen/MemSkill. Added by
scripts/apply_m8_token_patch.sh so SubMem's M8 substrate
(memory/m8_memskill_faithful.py) can recover token consumption from the
otherwise-opaque upstream subprocess.

Tokens are recorded per upstream *component* — an objective fact about
which part of the pipeline burned them:

  controller       — LLMController skill selection
  executor_write   — Executor LLM tokens attributed to INSERT actions
  executor_mgmt    — Executor LLM tokens attributed to UPDATE/DELETE actions
  designer         — Designer skill-bank evolution
  (judge is deliberately NOT instrumented — it is eval cost, not a
   memory operation, and must not pollute M8's E-metrics.)

The Executor makes one LLM call that can emit a mix of INSERT and
UPDATE/DELETE actions; record_executor() splits that call's tokens
between executor_write / executor_mgmt in proportion to the action
counts the parsed response yielded.

The M8 driver maps components to SubMem E-metric buckets (write =
controller + executor_write; mgmt = designer + executor_mgmt). Keeping
the component split here and the semantic mapping in the driver means a
change of definition only touches version-controlled SubMem code, not
this vendored patch.

At process exit (or on dump()) the totals are written to the JSON path
named by SUBMEM_M8_TOKEN_SIDECAR. Thread-safe: upstream's
llm_utils.request_task fans out over a ThreadPool.
"""
import atexit
import json
import os
import threading

_lock = threading.Lock()

_COMPONENTS = ("controller", "executor_write", "executor_mgmt", "designer")
_totals = {
    f"{c}_{k}": 0
    for c in _COMPONENTS
    for k in ("input", "output", "calls")
}


def _add(component, prompt_tokens, completion_tokens, calls=1):
    with _lock:
        _totals[f"{component}_input"] += int(prompt_tokens or 0)
        _totals[f"{component}_output"] += int(completion_tokens or 0)
        _totals[f"{component}_calls"] += int(calls)


def record_controller(prompt_tokens=0, completion_tokens=0):
    """One LLMController call."""
    _add("controller", prompt_tokens, completion_tokens)


def record_controller_usage(usage):
    """record_controller from an OpenAI `usage` object (may be None)."""
    if usage is None:
        return
    record_controller(
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def record_designer(prompt_tokens=0, completion_tokens=0):
    """One Designer LLM call."""
    _add("designer", prompt_tokens, completion_tokens)


def record_executor(prompt_tokens, completion_tokens,
                     n_insert, n_mgmt):
    """One Executor LLM call.

    The call's tokens are split between executor_write and executor_mgmt
    in proportion to action counts: n_insert INSERT actions vs n_mgmt
    UPDATE/DELETE actions. NOOP actions are ignored. If the call yielded
    no INSERT/UPDATE/DELETE at all, the whole call is attributed to
    executor_write (it was still a write-phase attempt).
    """
    n_insert = max(int(n_insert or 0), 0)
    n_mgmt = max(int(n_mgmt or 0), 0)
    total = n_insert + n_mgmt
    pt = int(prompt_tokens or 0)
    ct = int(completion_tokens or 0)
    if total == 0:
        _add("executor_write", pt, ct, calls=1)
        return
    w_frac = n_insert / total
    w_pt = round(pt * w_frac)
    w_ct = round(ct * w_frac)
    _add("executor_write", w_pt, w_ct, calls=1 if n_insert else 0)
    _add("executor_mgmt", pt - w_pt, ct - w_ct, calls=1 if n_mgmt else 0)


def snapshot():
    with _lock:
        return dict(_totals)


def dump():
    """Write the running totals to the sidecar path, if one is configured."""
    path = os.environ.get("SUBMEM_M8_TOKEN_SIDECAR", "").strip()
    if not path:
        return
    with _lock:
        data = dict(_totals)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass


# Flush on normal interpreter exit so M8 sees the totals even if the
# pipeline never calls dump() explicitly.
atexit.register(dump)
METER_EOF
echo "[patch] wrote src/_submem_token_meter.py"

# ── 2/3/4. Insert recorder calls into controller / executor / designer ──
# Each insertion is anchored on an exact upstream string and guarded by a
# marker so re-running is idempotent.
python3 - "$VENDOR" <<'PY_EOF'
import sys
vendor = sys.argv[1]


def patch(path, marker, anchor, insert, label):
    with open(path) as f:
        src = f.read()
    if marker in src:
        print(f"[patch] {label}: already patched, skipping")
        return
    if anchor not in src:
        print(f"[patch] ERR {label}: anchor not found — upstream changed?")
        sys.exit(1)
    src = src.replace(anchor, anchor + insert, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"[patch] {label}: instrumented")


def patch_replace(path, marker, old, new, label):
    with open(path) as f:
        src = f.read()
    if marker in src:
        print(f"[patch] {label}: already patched, skipping")
        return
    if old not in src:
        print(f"[patch] ERR {label}: anchor not found — upstream changed?")
        sys.exit(1)
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"[patch] {label}: instrumented")


# ── controller.LLMController: record the controller's own LLM call ──
patch(
    f"{vendor}/src/controller.py",
    marker="_submem_rc",
    anchor=(
        "            self.llm_input_tokens += int(getattr(usage, \"prompt_tokens\", 0) or 0)\n"
        "            self.llm_output_tokens += int(getattr(usage, \"completion_tokens\", 0) or 0)\n"
    ),
    insert=(
        "            # SubMem instrumentation patch: record token usage for M8.\n"
        "            try:\n"
        "                from src._submem_token_meter import record_controller_usage as _submem_rc\n"
        "                _submem_rc(usage)\n"
        "            except Exception:\n"
        "                pass\n"
    ),
    label="src/controller.py",
)

# ── executor.execute_operation: capture token return values ──
patch_replace(
    f"{vendor}/src/executor.py",
    marker="_exec_pt",
    old=(
        "                  response, _, _ = get_llm_response_via_api(\n"
        "                      prompt=instruction,\n"
    ),
    new=(
        "                  response, _exec_pt, _exec_ct = get_llm_response_via_api(\n"
        "                      prompt=instruction,\n"
    ),
    label="src/executor.py (capture tokens)",
)

# ── executor.execute_operation: split tokens after parse ──
patch_replace(
    f"{vendor}/src/executor.py",
    marker="_submem_re",
    old=(
        "            # Parse response (may contain multiple actions)\n"
        "            all_results.extend(self._parse_response(response, len(retrieved_memories)))\n"
    ),
    new=(
        "            # Parse response (may contain multiple actions)\n"
        "            _exec_results = self._parse_response(response, len(retrieved_memories))\n"
        "            all_results.extend(_exec_results)\n"
        "\n"
        "            # SubMem instrumentation patch: split this call's tokens between\n"
        "            # write (INSERT) and mgmt (UPDATE/DELETE) by action count.\n"
        "            try:\n"
        "                from src._submem_token_meter import record_executor as _submem_re\n"
        "                _n_ins = sum(1 for r in _exec_results\n"
        "                             if getattr(r, \"action_type\", \"\") == \"INSERT\")\n"
        "                _n_mgmt = sum(1 for r in _exec_results\n"
        "                              if getattr(r, \"action_type\", \"\") in (\"UPDATE\", \"DELETE\"))\n"
        "                _submem_re(_exec_pt, _exec_ct, _n_ins, _n_mgmt)\n"
        "            except Exception:\n"
        "                pass\n"
    ),
    label="src/executor.py (record split)",
)

# ── designer._call_llm_with_retry: capture + record the designer call ──
patch_replace(
    f"{vendor}/src/designer.py",
    marker="_dsg_pt",
    old=(
        "                response, _, _ = get_llm_response_via_api(\n"
        "                    prompt=prompt,\n"
        "                    LLM_MODEL=self.designer_model,\n"
        "                    base_url=self.args.api_base,\n"
        "                    api_key=self.args.api_key,\n"
        "                    MAX_TOKENS=max_tokens,\n"
        "                    TAU=tau\n"
        "                )\n"
        "                return response\n"
    ),
    new=(
        "                response, _dsg_pt, _dsg_ct = get_llm_response_via_api(\n"
        "                    prompt=prompt,\n"
        "                    LLM_MODEL=self.designer_model,\n"
        "                    base_url=self.args.api_base,\n"
        "                    api_key=self.args.api_key,\n"
        "                    MAX_TOKENS=max_tokens,\n"
        "                    TAU=tau\n"
        "                )\n"
        "                # SubMem instrumentation patch: record Designer token usage.\n"
        "                try:\n"
        "                    from src._submem_token_meter import record_designer as _submem_rd\n"
        "                    _submem_rd(_dsg_pt, _dsg_ct)\n"
        "                except Exception:\n"
        "                    pass\n"
        "                return response\n"
    ),
    label="src/designer.py",
)

# ── main.py: dump per-query predictions so the M8 adapter can read them ──
# Upstream only prints aggregate metrics and never persists predictions;
# the M8 substrate needs them written to args.out_file.
patch(
    f"{vendor}/main.py",
    marker="[submem] wrote",
    anchor=(
        "    if category_metrics:\n"
        "        print(\"\\nBy category:\")\n"
        "        for category in sorted(category_metrics.keys()):\n"
        "            data = category_metrics[category]\n"
        "            print(\n"
        "                f\"Category {category}: \"\n"
        "                f\"F1={_avg(data['f1']):.4f}, \"\n"
        "                f\"LLM Judge={_avg(data['llm_judge']):.4f}\"\n"
        "            )\n"
    ),
    insert=(
        "\n"
        "    # SubMem instrumentation patch: dump per-query predictions to\n"
        "    # args.out_file so the M8 substrate adapter can read answers back.\n"
        "    try:\n"
        "        _out = getattr(args, \"out_file\", None)\n"
        "        if _out:\n"
        "            _items = []\n"
        "            for _qid, _pred in predictions.items():\n"
        "                _meta = meta_by_qid.get(_qid, {})\n"
        "                _qa = _meta.get(\"qa\", {})\n"
        "                _items.append({\n"
        "                    \"question_id\": _qid,\n"
        "                    \"question\": _qa.get(\"question\", \"\"),\n"
        "                    \"prediction\": _pred,\n"
        "                    \"gold\": trainer.evaluator.get_ground_truth(_qa),\n"
        "                })\n"
        "            import os as _os\n"
        "            _os.makedirs(_os.path.dirname(_out) or \".\", exist_ok=True)\n"
        "            with open(_out, \"w\") as _f:\n"
        "                json.dump({\"results\": _items}, _f)\n"
        "            print(f\"[submem] wrote {len(_items)} predictions to {_out}\")\n"
        "    except Exception as _e:\n"
        "        print(f\"[submem] failed to write predictions: {_e}\")\n"
    ),
    label="main.py (predictions dump)",
)

# ── main.py: eval-path controller context + empty-bank guard ──
# The eval-only path (_extract_memory_actions_step) never populated the
# LLM controller's thread-local routing context, so the controller routed
# on stale state. It also offered delete/update even with an empty memory
# bank. This patch sets the context and restricts candidate operations to
# insert/noop when no memories were retrieved.
patch_replace(
    f"{vendor}/main.py",
    marker="SubMem empty-bank guard",
    old=(
        "    candidate_ops = trainer.operation_bank.get_candidate_operations()\n"
        "    op_embeddings = np.vstack([op.embedding for op in candidate_ops])\n"
        "\n"
        "    state_tensor = torch.tensor(state_embedding, dtype=torch.float32).to(trainer.device)\n"
        "    op_tensor = torch.tensor(op_embeddings, dtype=torch.float32).to(trainer.device)\n"
        "\n"
        "    action_idx, _, _ = trainer.controller(state_tensor, op_tensor, deterministic=True)\n"
    ),
    new=(
        "    candidate_ops = trainer.operation_bank.get_candidate_operations()\n"
        "    # SubMem empty-bank guard: when no memories were retrieved there\n"
        "    # is nothing to UPDATE or DELETE, so restrict candidate operations\n"
        "    # to insert/noop. Without this the LLM controller can pick\n"
        "    # \"delete\" on the first (empty-bank) chunk, the executor produces\n"
        "    # no actions, and the memory bank never fills. Filtered before\n"
        "    # op_embeddings/action_idx are derived so indices stay consistent.\n"
        "    try:\n"
        "        if not retrieved_memories:\n"
        "            _ins = [\n"
        "                op for op in candidate_ops\n"
        "                if str(getattr(op, \"update_type\", \"\")).strip().upper()\n"
        "                in (\"INSERT\", \"NOOP\")\n"
        "            ]\n"
        "            if _ins:\n"
        "                candidate_ops = _ins\n"
        "    except Exception:\n"
        "        pass\n"
        "    op_embeddings = np.vstack([op.embedding for op in candidate_ops])\n"
        "\n"
        "    state_tensor = torch.tensor(state_embedding, dtype=torch.float32).to(trainer.device)\n"
        "    op_tensor = torch.tensor(op_embeddings, dtype=torch.float32).to(trainer.device)\n"
        "\n"
        "    # SubMem patch: the LLM controller reads its routing context\n"
        "    # (chunk text, retrieved memories, candidate skills) from a\n"
        "    # thread-local channel — this eval path must populate it, or the\n"
        "    # controller routes on stale state and mis-picks skills.\n"
        "    try:\n"
        "        from src.controller import LLMController as _LLMC, llm_set_context as _llm_set_ctx\n"
        "        if isinstance(trainer.controller, _LLMC):\n"
        "            _llm_set_ctx(\n"
        "                session_text=session_text,\n"
        "                retrieved_memories=retrieved_memories,\n"
        "                candidate_ops=[\n"
        "                    {\"name\": op.name, \"description\": getattr(op, \"description\", \"\")}\n"
        "                    for op in candidate_ops\n"
        "                ],\n"
        "            )\n"
        "    except Exception:\n"
        "        pass\n"
        "\n"
        "    action_idx, _, _ = trainer.controller(state_tensor, op_tensor, deterministic=True)\n"
    ),
    label="main.py (eval-path controller context + empty-bank guard)",
)

# ── main.py: load an evolved skill bank from SUBMEM_M8_SKILL_BANK ──
# The default operation bank only seeds the 4 base ops. When the M8 driver
# exports SUBMEM_M8_SKILL_BANK (a JSON of operation templates built from
# skills/conversational_skills/*.md), replace the bank with that evolved
# set so M8 runs the paper-faithful trained MemSkill.
patch(
    f"{vendor}/main.py",
    marker="SUBMEM_M8_SKILL_BANK",
    anchor=(
        "    # Initialize trainer\n"
        "    print(\"\\nInitializing Agentic Memory Trainer...\")\n"
        "    trainer = get_trainer(args, config)\n"
    ),
    insert=(
        "\n"
        "    # SubMem patch: load an evolved skill bank from\n"
        "    # SUBMEM_M8_SKILL_BANK (a JSON of operation templates) if set,\n"
        "    # replacing the 4 base ops with the paper's trained skill bank.\n"
        "    try:\n"
        "        import os as _os, json as _json\n"
        "        _sb = _os.environ.get(\"SUBMEM_M8_SKILL_BANK\", \"\").strip()\n"
        "        if _sb and _os.path.isfile(_sb):\n"
        "            with open(_sb) as _f:\n"
        "                _tpl = _json.load(_f)\n"
        "            if _tpl:\n"
        "                trainer.operation_bank.load_from_templates(_tpl)\n"
        "                print(f\"[submem] loaded evolved skill bank: \"\n"
        "                      f\"{len(_tpl)} operations from {_sb}\")\n"
        "    except Exception as _e:\n"
        "        print(f\"[submem] failed to load evolved skill bank: {_e}\")\n"
    ),
    label="main.py (evolved skill bank loader)",
)
PY_EOF

echo "[patch] done — M8 token instrumentation applied to $VENDOR"
