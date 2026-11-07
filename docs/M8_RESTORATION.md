# M8 adapter restoration

## Source and scope

`memory/m8_memskill_faithful.py` is restored from commit
`3ef6ecf` of this repository. Commit `cafbfba` moved the file to
`memory/archived/` without changing its contents, but left the package initializer
importing its original path. Consequently, M8 was not registered and the factory
could not resolve either `M8` or its legacy alias `C20`.

The recovered source has SHA256
`8f46acec86be48e550393136597981deb72c85575a8cd483a343248f8e30e6a9`.
The archived copy is preserved as the historical reference. The restored active
copy adds dependency validation, explicit runtime errors, isolated temporary
split files, a timeout that also handles silent children, and answer-cache
invalidation. These are new engineering fixes; the restored file is not claimed
to be byte-identical to the code used for the paper's experiments.

`memory/m8_skill_bundles.py` remains a separate, legacy `C15` implementation.
It is not substituted for M8: its algorithm and read protocol differ.

## Missing runtime

The original customized `third_party/MemSkill` checkout was not tracked in this
repository and is currently unavailable. The historical adapter requires more
than the public [MemSkill source](https://github.com/ViktorAxelsen/MemSkill):

- `src.controller.LLMController` and `llm_set_context`, with trainer integration.
- `--controller-backend` and `--controller-llm-model` CLI options.
- Per-query JSON predictions written to `--out-file`.
- The `SUBMEM_M8_TOKEN_SIDECAR` token-meter integration.
- The `SUBMEM_M8_SKILL_BANK` evolved-skill-bank integration.

The existing `scripts/apply_m8_token_patch.sh` expects a previously customized
controller; it does not create that controller. The installer referenced by the
old adapter, `scripts/vendor_memskill.sh`, was not released. Cloning stock
upstream and running the token patch is therefore not a complete setup procedure.

If the original customized backend is recovered, place it at
`third_party/MemSkill` or set `MemoryConfig.extra["m8_vendor_dir"]` to its location.
The adapter checks for required files and controller/CLI symbols before use.
Those checks establish interface compatibility only, not historical provenance
or numerical reproducibility. Install that backend's dependencies in the same
Python environment as SubMem; the adapter uses the current interpreter.

The historical adapter batches staged dialogue questions, invokes upstream
evaluation, and returns final answers (`PARAMETRIC_OUTPUT=True`). Call
`stage_queries` before `read`; the existing benchmark hook does this. It is not
a replacement implementation of all agentic M8 experiment paths.

## Paper correspondence

The final paper describes online skill clustering, summarization, merging and
pruning, a bank capped at 12 skills, and all refined bundles supplied to the
zero-shot reader. The recovered adapter instead delegates to a missing customized
backend. Without that backend and the original run configuration, this repair
cannot establish that those paper-described behaviors were executed. No new
algorithm is presented here as historical experimental code, and no reported
scores have been changed or revalidated.

## Offline verification

Run from the repository root:

```bash
python -m unittest discover -s tests -p 'test_m8_restore.py' -v
```

The tests execute the real initializer, M8 adapter, base registry and ID mapping.
They stub unrelated substrate imports to avoid installing their GPU dependencies,
and use a deterministic fake backend process to check adapter I/O and failures.
No model downloads, network requests or paid API calls are required. Passing
these tests does not constitute a real MemSkill run or a reproduction of the
paper's results.
