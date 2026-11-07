"""Offline adapter contracts; this is not a MemSkill/model reproduction test."""

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_submem_m8_offline_test"

# Only unrelated substrates are stubbed. The package initializer, base class,
# M8 implementation and C20 -> M8 mapping below are the actual repository code.
OTHER_SUBSTRATES = {
    "m1_dense_vector": "DenseVectorMemory",
    "m2_sparse_vector": "SparseVectorMemory",
    "m3_summarized_text": "SummarizedTextMemory",
    "m4_amem": "AMemMemory",
    "m5_lightrag": "LightRAGMemory",
    "m6_hierarchical": "HierarchicalMemory",
    "m7_reasoning_bank": "ReasoningBankMemory",
    "m9_lora": "LoRAMemory",
    "m10_full_context": "FullContextMemory",
    "m11_epicache": "EpiCacheMemory",
}

FAKE_MAIN = r'''
# Required custom CLI markers: --controller-backend --controller-llm-model
import argparse
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data-file')
parser.add_argument('--out-file')
args, _ = parser.parse_known_args()
vendor = Path(__file__).resolve().parent
mode = json.loads((vendor / 'mode.json').read_text())['mode']
if mode == 'timeout':
    time.sleep(30)  # Deliberately silent; the parent must enforce its timeout.
if mode == 'failure':
    print('deterministic offline backend failure')
    sys.exit(7)
if mode == 'missing_output':
    sys.exit(0)
out = Path(args.out_file)
if mode == 'malformed_output':
    out.write_text('{not json')
    sys.exit(0)
records = json.loads(Path(args.data_file).read_text())
splits = json.loads(Path('data/longmemeval_s_splits.json').read_text())
assert splits == {'train': [], 'val': [], 'test': list(range(len(records)))}
assert len({row['question_id'] for row in records}) == len(records)
(out.parent / 'captured_input.json').write_text(json.dumps(records))
predictions = [
    {'question': row['question'],
     'prediction': row['question'] + '|' + str(sum(map(len, row['haystack_sessions'])))}
    for row in records
]
if mode == 'unanswered':
    predictions = []
out.write_text(json.dumps({'results': predictions}))
sidecar = {
    'controller_input': 2, 'controller_output': 3, 'controller_calls': 1,
    'executor_write_input': 5, 'executor_write_output': 7, 'executor_write_calls': 1,
    'designer_input': 11, 'designer_output': 13, 'designer_calls': 1,
    'executor_mgmt_input': 17, 'executor_mgmt_output': 19, 'executor_mgmt_calls': 1,
}
Path(os.environ['SUBMEM_M8_TOKEN_SIDECAR']).write_text(json.dumps(sidecar))
'''


class RestoredM8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stubs = {}
        for filename, classname in OTHER_SUBSTRATES.items():
            name = f"{PACKAGE}.memory.{filename}"
            stub = types.ModuleType(name)
            setattr(stub, classname, type(classname, (), {}))
            stubs[name] = stub
        cls.modules_patch = patch.dict(sys.modules, stubs)
        cls.modules_patch.start()
        spec = importlib.util.spec_from_file_location(
            PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = package
        spec.loader.exec_module(package)
        cls.memory = importlib.import_module(f"{PACKAGE}.memory")
        cls.adapter = importlib.import_module(f"{PACKAGE}.memory.m8_memskill_faithful")
        cls.config = importlib.import_module(f"{PACKAGE}.config")

    @classmethod
    def tearDownClass(cls):
        cls.modules_patch.stop()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="submem_m8_test_")
        self.addCleanup(self.temp.cleanup)
        self.vendor = Path(self.temp.name) / "vendor"
        (self.vendor / "src").mkdir(parents=True)
        (self.vendor / "data").mkdir()
        (self.vendor / "main.py").write_text(FAKE_MAIN)
        (self.vendor / "src/controller.py").write_text(
            "class LLMController: pass\ndef llm_set_context(**kwargs): pass\n"
        )
        (self.vendor / "src/config.py").write_text(
            "# --controller-backend --controller-llm-model\n"
        )
        (self.vendor / "src/trainer.py").write_text("# LLMController integration\n")
        (self.vendor / "src/_submem_token_meter.py").write_text("# offline fixture\n")
        self.splits = self.vendor / "data/longmemeval_s_splits.json"
        self.original_splits = '{"original_backend_split": true}\n'
        self.splits.write_text(self.original_splits)
        self.set_mode("success")
        env = patch.dict(os.environ, {
            "OPENAI_API_KEY": "offline-test-no-network",
            "M8_UPSTREAM_TIMEOUT": "1",
        })
        env.start()
        self.addCleanup(env.stop)

    def set_mode(self, mode):
        (self.vendor / "mode.json").write_text(json.dumps({"mode": mode}))

    def make_memory(self, memory_id="M8", **extra):
        config = self.config.MemoryConfig(
            memory_id=memory_id,
            extra={"m8_vendor_dir": str(self.vendor), **extra},
        )
        unit = self.memory.create_memory_unit(config, model=None)
        self.addCleanup(unit.reset)
        return unit

    def prepare(self, unit):
        unit.write("Alice visited Tokyo", {
            "speaker": "Alice", "session_id": "s1", "session_date": "2026-01-01"
        })
        unit.write("I remember", {"role": "assistant", "session_id": "s1"})
        unit.write("Alice returned", {"speaker": "Alice", "session_id": "s2"})
        unit.stage_queries([{"question": "Where?"}, {"question": "Who?"}])

    def test_real_factory_registers_m8_and_resolves_c20(self):
        for memory_id in ("M8", "C20"):
            with self.subTest(memory_id=memory_id):
                unit = self.make_memory(memory_id)
                self.assertIsInstance(unit, self.adapter.MemSkillFaithfulMemory)
                self.assertTrue(unit.PARAMETRIC_OUTPUT)
        self.assertNotIn("C15", self.memory.MEMORY_UNIT_REGISTRY)

    def test_missing_or_stock_backend_fails_before_run(self):
        missing = self.vendor / "absent"
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            self.make_memory(m8_vendor_dir=str(missing))
        (self.vendor / "src/controller.py").write_text("class PPOController: pass\n")
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            self.make_memory()

    def test_batch_translation_results_and_token_mapping(self):
        unit = self.make_memory()
        self.prepare(unit)
        self.assertEqual(unit.read("Where?")[0].content, "Where?|3")
        self.assertEqual(unit.read("Who?")[0].content, "Who?|3")
        records = json.loads((unit._scratch_dir / "captured_input.json").read_text())
        self.assertEqual([t["role"] for t in records[0]["haystack_sessions"][0]],
                         ["Alice", "assistant"])
        self.assertEqual(records[0]["haystack_dates"], ["2026-01-01", ""])
        self.assertEqual(len(records[0]["haystack_sessions"]), 2)
        self.assertNotIn("_session_id", records[0]["haystack_sessions"][0][0])
        tokens = unit.get_stats()["token_consumption"]
        self.assertEqual(tokens["write_llm_input"], 7)
        self.assertEqual(tokens["write_llm_output"], 10)
        self.assertEqual(tokens["mgmt_llm_input"], 28)
        self.assertEqual(tokens["mgmt_llm_output"], 32)
        self.assertEqual(tokens["read_calls"], 2)
        self.assertEqual(self.splits.read_text(), self.original_splits)

    def test_staging_is_required_and_cap_cannot_silently_drop_answers(self):
        unit = self.make_memory(m8_max_queries=1)
        with self.assertRaises(ValueError):
            unit.read("Where?")
        self.prepare(unit)
        with self.assertRaises(ValueError):
            unit.read("Who?")
        self.assertIsNone(unit._scratch_dir)

    def test_query_ids_are_unique_and_ambiguous_batches_are_rejected(self):
        unit = self.make_memory()
        unit.write("An observation")
        unit.stage_queries([
            {"id": "submem_q_1", "question": "First?"},
            {"question": "Second?"},
        ])
        unit.read("First?")
        records = json.loads((unit._scratch_dir / "captured_input.json").read_text())
        self.assertEqual(len({row["question_id"] for row in records}), 2)
        with self.assertRaises(ValueError):
            unit.stage_queries([
                {"id": "same", "question": "First?"},
                {"id": "same", "question": "Second?"},
            ])
        with self.assertRaises(ValueError):
            unit.stage_queries([{"question": "Same?"}, {"question": "Same?"}])

    def test_failed_backend_cannot_be_scored_as_empty_answers(self):
        for mode in ("failure", "missing_output", "malformed_output", "unanswered"):
            with self.subTest(mode=mode):
                self.set_mode(mode)
                unit = self.make_memory()
                self.prepare(unit)
                with self.assertRaises((RuntimeError, ValueError, FileNotFoundError)):
                    unit.read("Where?")
                self.assertFalse(unit._has_run)
                self.assertEqual(self.splits.read_text(), self.original_splits)

    def test_silent_child_obeys_timeout(self):
        self.set_mode("timeout")
        unit = self.make_memory()
        self.prepare(unit)
        started = time.monotonic()
        with self.assertRaises((RuntimeError, TimeoutError)):
            unit.read("Where?")
        self.assertLess(time.monotonic() - started, 6)
        self.assertFalse(unit._has_run)
        self.assertEqual(self.splits.read_text(), self.original_splits)

    def test_write_and_stage_queries_invalidate_answer_cache(self):
        unit = self.make_memory()
        self.prepare(unit)
        self.assertEqual(unit.read("Where?")[0].content, "Where?|3")
        unit.write("Another observation", {"session_id": "s2"})
        self.assertEqual(unit.read("Where?")[0].content, "Where?|4")
        unit.stage_queries([{"question": "New question?"}])
        self.assertEqual(unit.read("New question?")[0].content, "New question?|4")
        tokens = unit.get_stats()["token_consumption"]
        self.assertEqual(tokens["write_llm_input"], 21)
        self.assertEqual(tokens["mgmt_llm_input"], 84)
        self.assertEqual(tokens["read_calls"], 3)

    def test_reset_clears_state_and_removes_scratch(self):
        unit = self.make_memory()
        self.prepare(unit)
        unit.read("Where?")
        scratch = unit._scratch_dir
        unit.reset()
        self.assertFalse(scratch.exists())
        self.assertEqual(unit.size(), 0)
        self.assertFalse(unit._pending_queries)
        self.assertTrue(all(v == 0 for v in unit.get_stats()["token_consumption"].values()))

    def test_parallel_units_leave_vendor_split_untouched(self):
        units = [self.make_memory(), self.make_memory()]
        for unit in units:
            self.prepare(unit)
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = list(pool.map(lambda unit: unit.read("Where?")[0].content, units))
        self.assertEqual(answers, ["Where?|3", "Where?|3"])
        self.assertNotEqual(units[0]._scratch_dir, units[1]._scratch_dir)
        self.assertEqual(self.splits.read_text(), self.original_splits)


if __name__ == "__main__":
    unittest.main()
