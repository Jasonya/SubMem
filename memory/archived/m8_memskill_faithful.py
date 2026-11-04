"""M8 — Skill Bundles: MemSkill paper-faithful (vendored upstream + LLM controller).

This is the paper's M8 substrate. It vendors the full upstream
ViktorAxelsen/MemSkill pipeline at ``third_party/MemSkill`` and runs its
own ``main.py`` end-to-end via subprocess. The only deviation from
upstream is that the PPO controller is replaced by an LLM-API call
(see ``third_party/MemSkill/src/controller.py:LLMController``) — this is
the ``§`` footnote in the paper substrate table.

(A SubMem-native re-implementation of MemSkill with a zero-shot LLM
router also existed during development; it has been retired to
memory/archived/ — the paper reports this paper-faithful variant.)

This means:
  * **Executor**, **memory bank**, **operation bank**, **state encoder**
    (Longformer-base-4096), and **Designer** are upstream code, unmodified.
  * The PPO actor-critic at the controller call site is swapped for
    ``LLMController`` (gpt-4o-mini); call signatures are preserved.
  * Skill evolution (Designer) runs only when training cycles run.
    For eval-only runs it is dormant — same as upstream.

M8 does NOT plug into SubMem's BaseBenchmark.run() loop. Instead, the
adapter:

  1. Translates SubMem's BenchmarkSample list into LongMemEval JSON.
  2. Drops the JSON in a temp dir.
  3. Invokes ``python main.py --controller-backend llm ...`` from the
     vendored repo.
  4. Reads back the resulting per-query JSON and exposes it via the
     same MemoryEntry protocol the runner expects.

Because the underlying pipeline is paper-faithful, M8 numbers should
not be cross-compared to the other substrates (different retrieval
substrate, different encoder, different prompt).
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit


_REPO_ROOT = Path(__file__).resolve().parent.parent  # SubMem/
_VENDOR = _REPO_ROOT / "third_party" / "MemSkill"

# Upstream ships its trained ("evolved") skill bank as markdown files under
# third_party/MemSkill/skills/. conversational_skills/ is the dialog bank
# (9 skills: the 4 base ops + 5 Designer-evolved ones); embodied_task_skills/
# is the agentic bank. The default operation_bank only seeds the 4 base ops,
# so to run paper-faithful MemSkill we load the evolved bank from these .md.
_SKILL_BANK_DIR = _VENDOR / "skills" / "conversational_skills"


def _parse_skill_md(path: Path) -> Optional[Dict[str, Any]]:
    """Parse one upstream skill .md into an operation-template dict.

    The .md layout is: ``# Title`` then ``## Section`` blocks, ending with a
    line ``Action type: INSERT|UPDATE|DELETE|NOOP only.``. We map:
      name                 <- snake_cased title
      description          <- the ## Description section
      instruction_template <- everything from ## Purpose onward (the
                               operational guidance the executor needs)
      update_type          <- the Action type
    Returns None if the file lacks an Action type (not a valid skill).
    """
    text = path.read_text()
    lines = text.splitlines()
    title = ""
    for ln in lines:
        if ln.startswith("# "):
            title = ln[2:].strip()
            break
    name = (title or path.stem).lower().replace(" ", "_").replace("-", "_")

    # Section split on "## "
    sections: Dict[str, str] = {}
    cur = None
    buf: List[str] = []
    for ln in lines:
        if ln.startswith("## "):
            if cur is not None:
                sections[cur] = "\n".join(buf).strip()
            cur = ln[3:].strip().lower()
            buf = []
        elif cur is not None:
            buf.append(ln)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip()

    description = sections.get("description", title)

    import re as _re
    m = _re.search(r"Action type:\s*(INSERT|UPDATE|DELETE|NOOP)",
                   text, _re.IGNORECASE)
    if not m:
        return None
    update_type = m.group(1).lower()

    # instruction_template: from the Purpose section onward, verbatim, so the
    # executor sees the full operational guidance the way upstream templates do.
    pidx = text.find("## Purpose")
    instruction_template = text[pidx:].strip() if pidx >= 0 else text.strip()

    return {
        "name": name,
        "description": description,
        "instruction_template": instruction_template,
        "update_type": update_type,
        "meta_info": {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "evolved",
            "last_modified": "evolved",
        },
    }


def _build_evolved_skill_bank() -> Dict[str, Dict[str, Any]]:
    """Load all conversational skills from skills/*.md into a templates dict
    keyed by skill name — the format operation_bank.load_from_templates wants.
    """
    bank: Dict[str, Dict[str, Any]] = {}
    if not _SKILL_BANK_DIR.is_dir():
        return bank
    for md in sorted(_SKILL_BANK_DIR.glob("*.md")):
        tpl = _parse_skill_md(md)
        if tpl is not None:
            bank[tpl["name"]] = tpl
    return bank


@register_memory_unit("M8")
class MemSkillFaithfulMemory(BaseMemoryUnit):
    """Paper-faithful MemSkill via vendored upstream + LLM controller.

    Designed for dialog QA benchmarks (LongMemEval, LoCoMo). For agentic
    benchmarks (ALFWorld), upstream has its own pipeline; we don't wrap
    that here.

    Read returns the upstream-emitted answer directly — set
    PARAMETRIC_OUTPUT so the runner doesn't re-prompt on top.
    """

    PARAMETRIC_OUTPUT = True   # upstream eval emits final answer per query

    def __init__(self, config, model):
        super().__init__(config, model)
        if not _VENDOR.exists():
            raise FileNotFoundError(
                f"Upstream MemSkill not vendored at {_VENDOR}. "
                f"Run scripts/vendor_memskill.sh first.")

        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._controller_model = extra.get("m8_controller_llm_model", "gpt-4o-mini")
        self._controller_backend = extra.get("m8_controller_backend", "llm")
        self._chunk_size = extra.get("m8_chunk_size", 512)
        self._chunk_overlap = extra.get("m8_chunk_overlap", 64)
        self._session_mode = extra.get("m8_session_mode", "fixed-length")
        # For smoke testing, cap the number of queries that get sent to the
        # upstream pipeline. Upstream rebuilds the memory bank per question
        # (fc_sh_6k uses one shared conversation × 100 queries), so 100 is
        # ~99x more work than the conversation deserves. Default 5 lets us
        # validate the wiring before doing length sweeps.
        self._max_queries = extra.get("m8_max_queries", None)
        # Where to dump the run; cleaned up in reset() if temp.
        self._scratch_dir: Optional[Path] = None
        # Per-call accumulators
        self._buffer_sessions: List[List[Dict[str, str]]] = []
        self._buffer_dates: List[str] = []
        self._pending_writes: List[Dict[str, Any]] = []
        self._answers_by_qid: Dict[str, str] = {}
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_llm_input": 0, "read_llm_output": 0, "read_llm_calls": 0,
            "mgmt_llm_input": 0, "mgmt_llm_output": 0, "mgmt_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }
        # Per-component upstream token usage, recovered from the meter
        # sidecar (third_party/MemSkill/src/_submem_token_meter.py). Kept
        # separately from _token_counts so the raw component split stays
        # inspectable alongside the E-metric mapping. See _load_token_sidecar.
        self._upstream_token_components: Dict[str, int] = {}
        self._has_run = False

    # ─── BaseMemoryUnit interface ──────────────────────────────────────

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Buffer a turn. M8 batches all writes per-sample and flushes
        at the first read() call."""
        t0 = time.perf_counter()
        meta = metadata or {}
        # The upstream LongMemEval loader uses the turn's `role` purely as a
        # speaker label — it prepends ``f"{role}: {content}"`` to each chunk
        # (third_party/MemSkill/src/data_processing/longmemeval.py:111). So
        # `role` must carry the *real* speaker identity. LoCoMo turns have a
        # `speaker` name (e.g. "Caroline"/"Melanie") and no `role`; MAB turns
        # set `role` to user/assistant. Prefer role, fall back to the speaker
        # name — collapsing both speakers to "user" (the old behaviour)
        # destroyed the dialogue structure and produced empty answers.
        role = meta.get("role") or meta.get("speaker") or "user"
        # Heuristic: group consecutive turns into one "session" until we
        # see a new session_id. SubMem's MAB loader assigns sess_X_Y.
        session_id = meta.get("session_id", "default")
        if not self._buffer_sessions or self._buffer_sessions[-1][-1].get("_session_id") != session_id:
            self._buffer_sessions.append([])
            self._buffer_dates.append(meta.get("session_date", ""))
        self._buffer_sessions[-1].append({
            "role": role,
            "content": text,
            "_session_id": session_id,  # internal, stripped before write
        })
        self._write_times.append(time.perf_counter() - t0)

    def read(self, query: str, top_k: int = 10) -> List[MemoryEntry]:
        """First read triggers the upstream pipeline run; subsequent
        reads look up answers in the cache."""
        t0 = time.perf_counter()
        if not self._has_run:
            self._run_upstream()
            self._has_run = True
        # Look up answer keyed by query text (qid not always exposed up here).
        ans = self._answers_by_qid.get(query, "")
        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        return [MemoryEntry(
            content=ans, metadata={"source": "memskill_faithful"}, score=1.0,
        )]

    def reset(self):
        """Clear buffers; clean up scratch dir."""
        if self._scratch_dir is not None and self._scratch_dir.exists():
            try:
                shutil.rmtree(self._scratch_dir)
            except Exception:
                pass
        self._scratch_dir = None
        self._buffer_sessions.clear()
        self._buffer_dates.clear()
        self._pending_writes.clear()
        self._answers_by_qid.clear()
        self._write_times.clear()
        self._read_times.clear()
        self._has_run = False
        self._token_counts = {k: 0 for k in self._token_counts}
        self._upstream_token_components = {}
        self._pending_queries: List[Dict[str, str]] = []

    def size(self) -> int:
        return sum(len(s) for s in self._buffer_sessions)

    # SubMem's runner pre-loads all queries via the BenchmarkSample.queries
    # list before calling read() the first time, but our read() runs on
    # the very first call without seeing the future queries. Instead we
    # let the runner harvest queries via stage_queries() if available.
    def stage_queries(self, queries: List[Dict[str, Any]]):
        """Optional hook: SubMem's MAB loader can call this to register
        all queries before the first read() so we can run a single
        upstream pass."""
        self._pending_queries = list(queries)

    # ─── upstream pipeline invocation ─────────────────────────────────

    def _run_upstream(self):
        """Translate buffer -> LongMemEval JSON, run upstream main.py,
        parse results."""
        # Build LongMemEval-format data: one record per query, all sharing
        # the same haystack_sessions.
        # Strip internal _session_id helper key.
        sessions = [
            [{"role": t["role"], "content": t["content"]} for t in sess]
            for sess in self._buffer_sessions
        ]
        records = []
        queries = getattr(self, "_pending_queries", []) or []
        if self._max_queries is not None:
            queries = queries[: int(self._max_queries)]
            print(f"[M8] capped to {len(queries)} queries via m8_max_queries",
                  flush=True)
        if not queries:
            # No staged queries → can't drive upstream. Caller must call
            # stage_queries() or wrap this differently.
            self._answers_by_qid = {}
            return
        for q in queries:
            records.append({
                "question_id": q.get("id", ""),
                "question": q["question"],
                "answer": q.get("answer", ""),
                "question_date": q.get("question_date", ""),
                "haystack_sessions": sessions,
                "haystack_dates": self._buffer_dates,
            })

        # Dump to scratch dir
        scratch = Path(tempfile.mkdtemp(prefix="submem_m8_"))
        self._scratch_dir = scratch
        data_file = scratch / "lme_data.json"
        out_file = scratch / "results.json"
        save_dir = scratch / "save"
        log_dir = scratch / "log"
        save_dir.mkdir(); log_dir.mkdir()
        with open(data_file, "w") as f:
            json.dump(records, f)

        # Upstream main.split_data() reads ./data/longmemeval_s_splits.json
        # for LongMemEval and only puts indices listed under "test" into the
        # eval set. We need ALL our records to be in test, so we override
        # that file inside the vendored repo for the duration of this run.
        # Restore at the end via a sentinel file the cmd will delete via
        # `--data-file` cwd separation; simplest approach: write splits in
        # the vendored repo's data dir.
        splits_dir = _VENDOR / "data"
        splits_dir.mkdir(exist_ok=True)
        splits_file = splits_dir / "longmemeval_s_splits.json"
        # Backup if it existed
        backup = None
        if splits_file.exists():
            backup = splits_file.with_suffix(".json.submem_bak")
            splits_file.rename(backup)
        all_idx = list(range(len(records)))
        with open(splits_file, "w") as f:
            json.dump({"train": [], "val": [], "test": all_idx}, f)

        # Invoke upstream main.py
        env = os.environ.copy()
        api_key = os.environ.get("OPENAI_API_KEY", "")
        api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        env["OPENAI_API_KEY"] = api_key
        # SubMem token meter sidecar: the vendored pipeline's LLM call sites
        # (patched via src/_submem_token_meter) write process-wide token
        # totals here at exit. We read it back after the subprocess ends.
        token_sidecar = scratch / "token_usage.json"
        env["SUBMEM_M8_TOKEN_SIDECAR"] = str(token_sidecar)
        # Evolved skill bank: dump upstream's trained conversational skills
        # (skills/*.md) to JSON and point the patched main.py at it, so the
        # operation bank is the paper's evolved bank, not just the 4 base ops.
        skill_bank = _build_evolved_skill_bank()
        if skill_bank:
            skill_json = scratch / "evolved_skill_bank.json"
            with open(skill_json, "w") as f:
                json.dump(skill_bank, f)
            env["SUBMEM_M8_SKILL_BANK"] = str(skill_json)
            print(f"[M8] evolved skill bank: {len(skill_bank)} skills "
                  f"-> {skill_json}", flush=True)
        else:
            print("[M8] no evolved skill bank found; using base operations",
                  flush=True)
        if not api_key:
            print("[M8] WARNING: OPENAI_API_KEY not set; upstream will fail",
                  flush=True)
        # Upstream's --model defaults to a router model name (qwen3-next-80b)
        # that isn't a valid OpenAI model ID; the Executor/judge LLM calls
        # would fail with 'invalid model ID'. Force gpt-4o-mini (matches
        # what we use for the controller and for SubMem's LLM judge).
        upstream_llm_model = self._controller_model
        cmd = [
            "python", "main.py",
            "--dataset", "longmemeval",
            "--data-file", str(data_file),
            "--out-file", str(out_file),
            "--save-dir", str(save_dir),
            "--log-dir", str(log_dir),
            "--eval-only",
            "--controller-backend", self._controller_backend,
            "--controller-llm-model", self._controller_model,
            "--session-mode", self._session_mode,
            "--chunk-size", str(self._chunk_size),
            "--chunk-overlap", str(self._chunk_overlap),
            "--disable-flash-attn",  # flash_attn not installed in submem env
            # Upstream expects --api-key list and --api-base. Pass real values
            # so executor / judge / designer LLM calls don't hit placeholders.
            "--api", "--api-key", api_key, "--api-base", api_base,
            # Force the upstream Executor/judge/designer to real OpenAI model
            # IDs. Upstream defaults to a router name + 'openai/gpt-oss-120b'
            # which return 'invalid model ID' on api.openai.com.
            "--model", upstream_llm_model,
            "--llm-judge-model", upstream_llm_model,
        ]
        # Stream upstream stdout/stderr to our own stdout so SLURM log shows
        # progress; without this we'd buffer 100 query-iterations of output
        # in memory until the subprocess exits or times out.
        timeout_sec = int(os.environ.get("M8_UPSTREAM_TIMEOUT", "28800"))  # 8h default
        log_path = scratch / "upstream.log"
        print(f"[M8] launching upstream (timeout={timeout_sec}s); "
              f"streaming to {log_path} and stdout", flush=True)
        rc = -1
        try:
            with open(log_path, "w") as logf:
                proc = subprocess.Popen(
                    cmd, cwd=str(_VENDOR), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                t_start = time.time()
                for line in proc.stdout:
                    print(f"[M8.up] {line}", end="", flush=True)
                    logf.write(line)
                    if time.time() - t_start > timeout_sec:
                        proc.kill()
                        print(f"[M8] upstream timed out after {timeout_sec}s; killed",
                              flush=True)
                        # SIGKILL skips the meter's atexit dump, so the
                        # sidecar is usually absent here — read defensively.
                        self._load_token_sidecar(token_sidecar)
                        return
                rc = proc.wait()
        finally:
            # Restore upstream's splits.json regardless of success.
            # Guard against the backup having vanished (e.g. a concurrent
            # M8 run touching the same vendored file, or a crashed prior
            # run): only rename if the backup still exists, otherwise just
            # remove our temporary splits so the dir is left clean.
            if backup is not None and backup.exists():
                if splits_file.exists():
                    splits_file.unlink()
                backup.rename(splits_file)
            else:
                if splits_file.exists():
                    splits_file.unlink()
        if rc != 0:
            print(f"[M8] upstream main.py failed (rc={rc}). See {log_path}",
                  flush=True)
            # Subprocess exited normally (non-zero rc), so the meter's
            # atexit dump ran — recover whatever tokens were billed.
            self._load_token_sidecar(token_sidecar)
            return

        # Recover token usage from the meter sidecar regardless of whether
        # the pipeline produced answers — the LLM calls were billed either
        # way. The meter reports a process-wide total (controller + executor
        # + designer + judge) that cannot be split into write/read/mgmt.
        self._load_token_sidecar(token_sidecar)

        # Parse output: upstream emits per-query JSON with .question and
        # .generated/.answer fields.
        if not out_file.exists():
            print(f"[M8] expected output {out_file} not produced")
            return
        with open(out_file) as f:
            out = json.load(f)
        # Defensive: out may be {"results": [...]}
        items = out.get("results", out) if isinstance(out, dict) else out
        for item in items:
            q = item.get("question", "")
            a = item.get("generated") or item.get("answer_pred") or item.get("prediction") or ""
            if q:
                self._answers_by_qid[q] = a

    def _load_token_sidecar(self, sidecar_path):
        """Read the upstream token-meter sidecar and map upstream components
        to SubMem E-metric buckets.

        The meter (third_party/MemSkill/src/_submem_token_meter.py) reports
        tokens per upstream component. We map them to SubMem's write/mgmt
        buckets — this mapping is a SubMem modelling choice, kept here in
        version-controlled code rather than baked into the vendor patch:

          write (E5/E6 construct)  = controller + executor_write
          mgmt  (E10/E11)          = designer  + executor_mgmt
          read  (E8/E9)            = 0  (the upstream pipeline does its
                                         retrieval + answer generation
                                         inside the run; no separable
                                         per-query read LLM call is
                                         attributed here)
          judge                    = excluded (eval cost, not instrumented)
        """
        try:
            if not sidecar_path.exists():
                print(f"[M8] token sidecar not found at {sidecar_path}; "
                      "upstream token usage unavailable", flush=True)
                return
            with open(sidecar_path) as f:
                comp = json.load(f)
            self._upstream_token_components = dict(comp)

            def _g(key):
                return int(comp.get(key, 0) or 0)

            # write = controller + executor_write
            self._token_counts["write_llm_input"] = (
                _g("controller_input") + _g("executor_write_input"))
            self._token_counts["write_llm_output"] = (
                _g("controller_output") + _g("executor_write_output"))
            self._token_counts["write_llm_calls"] = (
                _g("controller_calls") + _g("executor_write_calls"))
            # mgmt = designer + executor_mgmt
            self._token_counts["mgmt_llm_input"] = (
                _g("designer_input") + _g("executor_mgmt_input"))
            self._token_counts["mgmt_llm_output"] = (
                _g("designer_output") + _g("executor_mgmt_output"))
            self._token_counts["mgmt_llm_calls"] = (
                _g("designer_calls") + _g("executor_mgmt_calls"))

            print(f"[M8] upstream token usage mapped to E-metrics: "
                  f"write(in/out/calls)="
                  f"{self._token_counts['write_llm_input']}/"
                  f"{self._token_counts['write_llm_output']}/"
                  f"{self._token_counts['write_llm_calls']}  "
                  f"mgmt="
                  f"{self._token_counts['mgmt_llm_input']}/"
                  f"{self._token_counts['mgmt_llm_output']}/"
                  f"{self._token_counts['mgmt_llm_calls']}", flush=True)
        except Exception as e:
            print(f"[M8] failed to read token sidecar: {e}", flush=True)

    # ─── stats hook ───────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        s = super().get_stats()
        s.update({
            "controller_backend": self._controller_backend,
            "controller_llm_model": self._controller_model,
            "token_consumption": dict(self._token_counts),
            # Raw per-component upstream token split, before the E-metric
            # mapping in _load_token_sidecar — kept for auditability.
            "upstream_token_components": dict(self._upstream_token_components),
        })
        return s
