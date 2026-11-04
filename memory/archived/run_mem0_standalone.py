"""Standalone Mem0 evaluation on LoCoMo — faithful to Mem0's eval code.

Copies Mem0's exact pipeline (dual-speaker add/search, prompts, GPT-4o-mini)
but uses local mem0.Memory instead of MemoryClient cloud API.

Output: results JSON compatible with our metrics pipeline.

Usage:
    python run_mem0_standalone.py --max-samples 1
    python run_mem0_standalone.py --max-samples 10 --is-graph
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Auto-load .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from jinja2 import Template
from openai import OpenAI
from mem0 import Memory

# ============================================================
# Prompts (exact copy from Mem0 evaluation/prompts.py)
# ============================================================
ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.),
   calculate the actual date based on the memory timestamp. For example, if a memory from
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example,
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character
   names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}

Answer:
"""

# Custom instructions (exact copy from Mem0 evaluation/src/memzero/add.py)
CUSTOM_INSTRUCTIONS = """
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
   - Name specific activities
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure
"""


def create_mem0():
    """Create local Mem0 Memory instance (replaces MemoryClient)."""
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.0,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            }
        },
        "custom_prompt": CUSTOM_INSTRUCTIONS,
        "version": "v1.1",
    }
    return Memory.from_config(config)


def add_memories(mem0, conversation, idx, batch_size=2):
    """Add memories for both speakers (faithful to Mem0 eval add.py)."""
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    speaker_a_id = f"{speaker_a}_{idx}"
    speaker_b_id = f"{speaker_b}_{idx}"

    # Delete existing
    try:
        mem0.delete_all(user_id=speaker_a_id)
        mem0.delete_all(user_id=speaker_b_id)
    except Exception:
        pass

    t0 = time.time()
    total_calls = 0

    for key in conversation.keys():
        if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
            continue

        date_time_key = key + "_date_time"
        timestamp = conversation.get(date_time_key, "")
        chats = conversation[key]

        # Build messages for both perspectives
        messages = []
        messages_reverse = []
        for chat in chats:
            if chat["speaker"] == speaker_a:
                messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
            elif chat["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})

        # Add in batches — speaker A
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            try:
                mem0.add(batch, user_id=speaker_a_id, metadata={"timestamp": timestamp})
                total_calls += 1
            except Exception as e:
                pass

        # Add in batches — speaker B
        for i in range(0, len(messages_reverse), batch_size):
            batch = messages_reverse[i:i + batch_size]
            try:
                mem0.add(batch, user_id=speaker_b_id, metadata={"timestamp": timestamp})
                total_calls += 1
            except Exception as e:
                pass

        print(f"    Session {key}: {len(chats)} turns added for both speakers", flush=True)

    elapsed = time.time() - t0
    print(f"  Add complete: {total_calls} API calls in {elapsed:.0f}s", flush=True)
    return speaker_a_id, speaker_b_id


def search_and_answer(mem0, openai_client, speaker_a_id, speaker_b_id, question, top_k=10, model="gpt-4o-mini"):
    """Search both speakers and generate answer (faithful to Mem0 eval search.py)."""
    # Search speaker A
    try:
        results_a = mem0.search(query=question, user_id=speaker_a_id, limit=top_k)
        if isinstance(results_a, dict):
            results_a = results_a.get("results", [])
    except Exception:
        results_a = []

    # Search speaker B
    try:
        results_b = mem0.search(query=question, user_id=speaker_b_id, limit=top_k)
        if isinstance(results_b, dict):
            results_b = results_b.get("results", [])
    except Exception:
        results_b = []

    # Format memories with timestamps
    def format_memories(results):
        formatted = []
        for r in results:
            if isinstance(r, dict):
                mem = r.get("memory", "")
                meta = r.get("metadata") or {}
            elif hasattr(r, "memory"):
                mem = r.memory or ""
                meta = getattr(r, "metadata", None) or {}
            else:
                continue
            ts = meta.get("timestamp", "") if isinstance(meta, dict) else ""
            formatted.append(f"{ts}: {mem}" if ts else mem)
        return formatted

    mem_a = format_memories(results_a)
    mem_b = format_memories(results_b)

    # Build prompt (exact Mem0 template)
    template = Template(ANSWER_PROMPT)
    prompt = template.render(
        speaker_1_user_id=speaker_a_id.split("_")[0],
        speaker_2_user_id=speaker_b_id.split("_")[0],
        speaker_1_memories=json.dumps(mem_a, indent=4),
        speaker_2_memories=json.dumps(mem_b, indent=4),
        question=question,
    )

    # Generate answer with GPT-4o-mini
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
    )
    answer = response.choices[0].message.content.strip()

    return answer, mem_a, mem_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Load data
    data_path = os.path.expanduser("~/.cache/submem/datasets/locomo/locomo10.json")
    with open(data_path) as f:
        data = json.load(f)

    if args.max_samples:
        data = data[:args.max_samples]

    output_dir = args.output_dir or os.environ.get(
        "SUBMEM_RESULTS",
        os.path.join(os.path.dirname(__file__), "results")
    )
    run_dir = os.path.join(output_dir, f"mem0_standalone_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    mem0 = create_mem0()
    openai_client = OpenAI()

    all_results = []

    for idx, item in enumerate(data):
        conversation = item["conversation"]
        qa = item["qa"]
        print(f"\n{'='*60}")
        print(f"  Conversation {idx}: {conversation['speaker_a']} & {conversation['speaker_b']}")
        print(f"  {len(qa)} questions")
        print(f"{'='*60}")

        # Add memories
        speaker_a_id, speaker_b_id = add_memories(mem0, conversation, idx)

        # Answer questions
        for qi, q in enumerate(qa):
            question = q["question"]
            gold = q.get("answer", "")
            category = q.get("category", 0)

            answer, mem_a, mem_b = search_and_answer(
                mem0, openai_client, speaker_a_id, speaker_b_id,
                question, top_k=args.top_k, model=args.model,
            )

            result = {
                "conversation_idx": idx,
                "query_id": f"{item['sample_id']}_q{qi}",
                "query": question,
                "gold_answer": gold,
                "prediction": answer,
                "query_type": {1: "single_hop", 2: "temporal", 3: "multi_hop",
                               4: "open_domain", 5: "adversarial"}.get(category, "unknown"),
                "category": category,
                "speaker_1_memories": mem_a,
                "speaker_2_memories": mem_b,
                "adversarial_answer": q.get("adversarial_answer", ""),
                "evidence": q.get("evidence", []),
            }
            all_results.append(result)

            if (qi + 1) % 20 == 0:
                print(f"    [{qi+1}/{len(qa)}] answered", flush=True)

        print(f"  Done: {len(qa)} questions answered")

    # Save results
    output_file = os.path.join(run_dir, "mem0_predictions.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Quick eval
    from collections import Counter
    import re, string

    def normalize(text):
        text = str(text).lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())

    def f1(pred, gold):
        pt = set(normalize(pred).split())
        gt = set(normalize(gold).split())
        if not gt: return 1.0 if not pt else 0.0
        c = pt & gt
        if not c: return 0.0
        p, r = len(c)/len(pt), len(c)/len(gt)
        return 2*p*r/(p+r)

    by_type = defaultdict(list)
    all_f1 = []
    for r in all_results:
        if r["category"] == 5:
            continue
        score = f1(r["prediction"], str(r["gold_answer"]))
        all_f1.append(score)
        by_type[r["query_type"]].append(score)

    print(f"\n{'='*60}")
    print(f"  Mem0 Standalone Results (model={args.model}, top_k={args.top_k})")
    print(f"{'='*60}")
    print(f"  Overall F1: {sum(all_f1)/len(all_f1)*100:.1f} (n={len(all_f1)})")
    for qt in ["single_hop", "multi_hop", "temporal", "open_domain"]:
        scores = by_type.get(qt, [])
        if scores:
            print(f"  {qt:<18s} F1={sum(scores)/len(scores)*100:.1f} (n={len(scores)})")


if __name__ == "__main__":
    main()
