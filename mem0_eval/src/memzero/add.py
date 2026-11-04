import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from mem0 import Memory

# ── Adapter: local Memory replaces MemoryClient ──
def _create_local_memory(custom_instructions, model="gpt-4o-mini"):
    import os as _os
    # Route qdrant storage to our per-PID MEM0_DIR (default /tmp/qdrant collides
    # across concurrent sweep jobs and is wiped on node reboot — both cause
    # search to return empty on subsequent reads).
    mem0_dir = _os.environ.get("MEM0_DIR") or _os.path.join(_os.path.expanduser("~"), ".mem0")
    # Use ChromaDB instead of qdrant-local to avoid qdrant's
    # payload_mask vs deleted-mask shape-mismatch bug that surfaces after
    # a few hundred UPDATE/DELETE operations. ChromaDB rebuilds ID maps
    # per operation and tolerates churn.
    # Default = qdrant (preserves LoCoMo/LongMemEval reproducibility of existing
    # runs). Opt in to chromadb via MEM0_VECTOR_STORE=chromadb when running
    # workloads that trigger qdrant's UPDATE/DELETE shape-mismatch bug
    # (e.g. ALFWorld with 200+ trajectories).
    # Mem0 2.x uses "chroma" (not "chromadb") as the provider name.
    raw = _os.environ.get("MEM0_VECTOR_STORE", "qdrant")
    vector_backend = "chroma" if raw in ("chromadb", "chroma") else raw
    history_db_path = _os.path.join(mem0_dir, "history.db")
    if vector_backend == "chroma":
        chroma_path = _os.path.join(mem0_dir, "chroma")
        vector_config = {
            "provider": "chroma",
            "config": {"path": chroma_path, "collection_name": "mem0"},
        }
    else:
        qdrant_path = _os.path.join(mem0_dir, "qdrant")
        vector_config = {
            "provider": "qdrant",
            "config": {"path": qdrant_path, "on_disk": True},
        }
    config = {
        "llm": {"provider": "openai", "config": {"model": model, "temperature": 0.0}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
        "vector_store": vector_config,
        "history_db_path": history_db_path,
        "custom_prompt": custom_instructions,
        "version": "v1.1",
    }
    return Memory.from_config(config)


# Update custom instructions
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


class MemoryADD:
    def __init__(self, data_path=None, batch_size=2, is_graph=False, mem0=None, model="gpt-4o-mini"):
        # Adapter: accept pre-built Memory instance or create local one
        self.mem0_client = mem0 or _create_local_memory(custom_instructions, model=model)
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        # Per-call latency tracking — surfaces E3 p50/p90 (was None when only
        # the aggregated total_add_time was recorded).
        if not hasattr(self, "add_call_latencies_ms"):
            self.add_call_latencies_ms = []
        for attempt in range(retries):
            try:
                t0 = time.perf_counter()
                _ = self.mem0_client.add(
                    message, user_id=user_id, metadata=metadata
                )
                self.add_call_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc="", dia_ids=None):
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]
            metadata = {"timestamp": timestamp}
            # Attach source dia_ids so retrieved memories can be matched to gold evidence
            if dia_ids:
                batch_dia_ids = dia_ids[i : i + self.batch_size]
                metadata["dia_ids"] = [d for d in batch_dia_ids if d]
            self.add_memory(speaker, batch_messages, metadata=metadata)

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # delete all memories for the two users
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages = []
            messages_reverse = []
            dia_ids = []
            for chat in chats:
                dia_id = chat.get("dia_id", "")
                dia_ids.append(dia_id)
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # add memories for the two users on different threads
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
                kwargs={"dia_ids": dia_ids},
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B"),
                kwargs={"dia_ids": dia_ids},
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

            for future in futures:
                future.result()
