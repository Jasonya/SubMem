# SubMem Prompt Reference

Systematic index of all prompts used across C1–C14 memory substrates and benchmarks.

---

## 1. Answer Generation Prompts (Benchmark-level)

These are the final prompts sent to the LLM to generate an answer, shared across all C* methods (except parametric C8/C9 which skip retrieval).

### LoCoMo (`benchmarks/locomo.py:121`)

```
You are an intelligent memory assistant tasked with retrieving accurate
information from conversation memories.

# CONTEXT:
You have access to memories from speakers in a conversation. These memories
contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence
4. If memories contain contradictory information, prioritize the most recent
5. Convert relative time references to absolute dates (e.g., "last year" → 2021)
6. Always convert relative time references to specific dates/months/years
7. Focus only on memory content. Do not confuse character names with actual users
8. Answer should be less than 5-6 words
9. If NO evidence for the event/fact in question, answer with empty string

# APPROACH (Think step by step):
1–8. [step-by-step reasoning chain]

Memories:
{retrieved_context}

Question: {query.question}

Answer:
```

### LoCoMo — Mem0 variant (`run_mem0_direct.py:76`)

Same as above but uses **Jinja2 template** with `{{speaker_1_memories}}` / `{{speaker_2_memories}}` split by speaker. Also adds an adversarial check (instruction 9 in locomo.py, absent in Mem0 original).

### LongMemEval (`benchmarks/longmemeval.py:109`)

```
You are a helpful assistant with access to conversation memory.

Retrieved memory context:
{retrieved_context}

Question: {query.question}

Answer the question based on the memory context.
If the information has been updated, provide the most recent version.
If you are not sure, say 'I don't know'.
Answer:
```

### Parametric (C8/C9)

No answer prompt — direct forward pass on the edited/fine-tuned model:
```
[parametric] {question_text}
```

---

## 2. Write-Phase Prompts (per substrate)

### C1 (Dense Vector) — `c1_dense_vector.py`
No LLM prompts. Write = encode utterance → FAISS add.

### C2 (Sparse Vector) — `c2_sparse_vector.py`
No LLM prompts. Write = tokenize → BM25 corpus add.

### C3 (Raw Text / MemGPT) — `c3_raw_text.py`

**Core Memory Update** (line 39):
```
You are managing an agent's core memory — a structured store of key facts.
After reading the new messages, decide what to ADD or REPLACE in core memory.

Current core memory:
<human>{human_block}</human>
<persona>{persona_block}</persona>

New messages to process:
{new_messages}

Instructions:
- <human> stores key facts: names, preferences, relationships, dates, goals
- <persona> stores agent's role and conversation context
- Only store important, reusable facts
- Output: APPEND/REPLACE human/persona: <text>
- If no changes: NONE

Changes:
```

**Eviction Summary** (line 68):
```
Summarise the following conversation messages into a concise paragraph (≤300 words).
Preserve key facts, dates, names, decisions, and any information the user would want remembered.

Messages:
{messages}

Summary:
```

### C4 (Summarized Text / ReadAgent) — `c4_summarized_text.py`

**Gist Generation** (line 24):
```
Please shorten the following conversation passage into a concise gist paragraph.
Keep key facts: who, what, when, and relationships. Just give me a shortened version.
Do NOT explain your reasoning.

Passage:
{chunk}

Shortened version:
```

### C5 (Extracted Facts / Mem0) — `run_mem0_direct.py:120`

**Custom Instructions** (passed to Mem0 library):
```
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name (not "user")
   - Personal details, emotional states, ongoing journeys
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity journeys, family planning, creative outlets
   - Mental health, career aspirations, life events

3. Make each memory rich with specific details
   - Include timeframes (exact dates when possible)
   - Name specific activities
   - Include emotional context

4. Extract memories only from user messages, not assistant responses

5. Format each memory as a paragraph with clear narrative structure
```

(Mem0 internally uses its own fact extraction + ADD/UPDATE/DELETE/NOOP classification prompts.)

### C6-KG (Temporal Knowledge Graph) — `c6_knowledge_graph.py`

**Entity Extraction** (line 87):
```
You are an expert entity extractor. Extract all significant entities from the current message.
Use previous messages ONLY for context. Extract entities ONLY from the current message.

Rules:
- ALWAYS extract the speaker/actor as the first entity
- Use FULL QUALIFIED names ("Nisha's dad" not "dad")
- EXCLUDE: pronouns, very generic words
- INCLUDE: people, places, organizations, activities, events, topics
- Types: Person, Location, Organization, Event, DateTime, Concept, Object, Other

Output: ENTITY_NAME | ENTITY_TYPE

Previous messages (for context only):
{previous_context}

Current message:
{chunk}

Entities:
```

**Entity Resolution** (line 109):
```
Determine if the NEW ENTITY is a duplicate of any EXISTING ENTITY.
Entities are duplicates ONLY if they refer to the same real-world object.

Duplicates: same entity in different contexts, abbreviations/aliases, synonymous references
NOT duplicates: related but distinct, same name but different referent

New entity: {new_entity} ({new_type})
Existing entities: {existing_str}

If duplicate: DUPLICATE: <existing_entity_name>
If not: NEW
```

**Edge (Fact) Extraction** (line 140):
```
You are an expert fact extractor. Extract factual relationships from the current message.
Use previous messages ONLY for context.

Rules:
- Each fact: TWO DISTINCT entities from the entity list
- Replace pronouns with actual entity names
- RELATION_TYPE: SCREAMING_SNAKE_CASE (LIVES_IN, WORKS_AT, HAPPENED_ON)
- FACT: detailed natural language statement
- Temporal: ISO 8601 dates when mentioned; "ongoing" / "none"

Output: SOURCE_ENTITY | RELATION_TYPE | TARGET_ENTITY | FACT | VALID_AT | INVALID_AT

Reference time: {reference_time}
Previous messages: {previous_context}
Current message: {chunk}
Entities: {entities_str}

Facts:
```

**Edge Conflict Resolution** (line 168):
```
You are a fact-checking assistant. Compare NEW FACT against EXISTING FACTS.

Determine:
1. DUPLICATE? (identical factual information)
2. CONTRADICT? (conflicts semantically)

IMPORTANT:
- NEVER mark facts with key differences in numbers/dates/qualifiers as duplicates
- Different events on different days → neither duplicates nor contradictions

Existing facts: {existing_str}
New fact: {new_fact}

Respond: DUPLICATE: <idx> / CONTRADICTS: <idx> / NEW
```

**Query Entity Extraction** (line 131):
```
Extract the key entities mentioned in the following question.
Output one entity name per line, no numbering, no explanation.
If no entities found, output NONE.

Question: {query}

Entities:
```

### C6-Graphiti — `c6_graphiti.py`
No custom prompts — uses `graphiti-core` library's built-in LLM extraction pipeline.

### C7 (Hierarchical / RAPTOR) — `c7_hierarchical.py`

**Cluster Summarization** (line 230):
```
Summarize the following conversation segments into a concise summary that captures
all key facts, events, preferences, and relationships mentioned. Preserve specific
details like names, dates, locations, and opinions.

Conversation segments:
{context}

Concise summary:
```

### C8 (MEMIT) — `c8_memit.py`

**Triple Extraction** (line 28):
```
Extract factual knowledge from the following conversation as (subject, relation, target) triples.

Rules:
- Subject: the entity the fact is about
- Relation: short phrase (e.g., "lives in", "works at", "has pet named")
- Target: the value/object of the fact
- One triple per line: SUBJECT | RELATION | TARGET
- Only clearly stated facts, not opinions or speculation
- Skip greetings and filler
- If no facts: NONE

Conversation:
{text}

Triples:
```

### C9 (LoRA) — `c9_lora.py`

**QA Generation** (line 27):
```
Given the following conversation, generate question-answer pairs that capture the key facts.

Rules:
- Each QA pair tests a specific fact (who, what, where, when)
- Answers should be short (a few words)
- One pair per line: Q: <question> | A: <answer>
- Generate as many pairs as there are distinct facts
- IMPORTANT: Convert relative time references to absolute dates using reference date
- Skip greetings and filler

Reference date: {reference_date}

Conversation:
{text}

QA pairs:
```

### C10 (Full Context) — `c10_full_context.py`
No LLM prompts. Write = append to buffer. Read = concatenate + truncate to context window.

### C11 (SnapKV) — `c11_snapkv.py`
No LLM prompts. Write = buffer text. Read = attention-guided KV cache compression.

### C12 (EpiCache) — `c12_epicache.py`
No LLM prompts. Write = buffer + segment + cluster. Read = episodic retrieval via embeddings.

### C13 (Sleep-time Compute) — `c13_sleep_time.py`

**Initial Rethink** (line 46):
```
You are an expert reasoning agent. Your task is to think about a conversation
passage and pre-compute useful information for answering future questions about it.

Given the passage below, produce a comprehensive "rethink memory block" containing:
1. Key facts and entities (names, dates, places, numbers)
2. Relationships between people/entities
3. Temporal details and sequence of events
4. Inferences that can be drawn
5. Potential questions and pre-computed answers
6. Calculations, cross-references, or reasoning for quick answering

Be thorough and verbose. Include everything that could be useful.

Passage:
{chunk}

Rethink memory block:
```

**Refinement Round** (line 66):
```
You are an expert reasoning agent refining a pre-computed memory block.

Below is a conversation passage and your previous rethink memory block. Your job:
1. Double-check all facts, inferences, and calculations for correctness
2. Add any missing information, relationships, or potential questions
3. Remove anything unhelpful or redundant
4. Make the block more useful for answering future questions

IMPORTANT: Output the COMPLETE refined memory block (fully replaces previous).

Passage:
{chunk}

Previous rethink memory block:
{previous_block}

Refined rethink memory block:
```

### C14 (ReasoningBank / Strategy Distillation) — `c14_reasoning_bank.py`

**Distillation** (line 46):
```
You are an expert at analyzing conversations. You will be given a passage from
a multi-session conversation between people.

## Guidelines
Extract the most important information as structured memory items.

## What to extract
- Key facts: names, dates, locations, numbers, specific events
- Relationships: who knows whom, family ties, work connections
- Changes over time: updates to plans, opinion changes, moved locations
- Reasoning insights: why something happened, cause-and-effect, contradictions

## Important notes
- At most {max_items} memory items per passage
- No overlapping items
- Always include specific names, dates, details
- Each Content field must be self-sufficient (no need for original passage)

## Output Format (Markdown)
# Memory Item 1
## Title <concise topic with key entity names>
## Description <one sentence with the most important fact>
## Content <2-4 sentences: who, what, when, where, why>

Conversation passage:
{chunk}

Memory items:
```

---

## 3. Read-Phase Prompts (per substrate)

### C3 (MemGPT) — Search Decision (line 78):
```
You are a memory retrieval agent. Given a user question and context, decide what to search.

Core memory:
<human>{human_block}</human>

Recent messages:
{recent}

User question: {query}

Two search tools:
1. archival_memory_search: semantic search over stored knowledge
2. conversation_search: hybrid search for exact phrases, recent context

Respond with ONE line:
- ARCHIVAL: <search query>
- RECALL: <search query>
- BOTH: <search query>
- NONE (answer in core memory or recent messages)
```

### C4 (ReadAgent) — Page Selection (line 33):
```
Below is a list of gist summaries of conversation pages.
Based on the question, decide which pages to look up for the full text.
Return ONLY the page numbers separated by commas (e.g., "3, 7, 12").
If unsure, pick the most relevant 1-3 pages.

Gists:
{gist_list}

Question: {query}

Pages to look up:
```

### C6-KG — Query Entity Extraction (see Write section above, line 131)

### All others
Read-phase uses embedding similarity (C1, C5, C13, C14), BM25 (C2), attention scoring (C11), clustering (C12), or direct generation (C8, C9, C10) — no LLM prompts.

---

## 4. Summary Table

| C | Write Prompts | Read Prompts | Answer Prompt |
|---|---------------|--------------|---------------|
| C1 | — | — | benchmark |
| C2 | — | — | benchmark |
| C3 | core_memory_update, eviction_summary | search_decide | benchmark |
| C4 | gist | page_select | benchmark |
| C5 | custom_instructions (Mem0 internal) | — | mem0 answer_prompt |
| C6-KG | entity, entity_resolve, edge, edge_resolve | query_entity | benchmark |
| C6-Graphiti | (graphiti-core internal) | (graphiti-core internal) | benchmark |
| C7 | summarize | — | benchmark |
| C8 | triple_extraction | — | [parametric] |
| C9 | qa_generation | — | [parametric] |
| C10 | — | — | benchmark |
| C11 | — | — | benchmark |
| C12 | — | — | benchmark |
| C13 | rethink_initial, rethink_refine | — | benchmark |
| C14 | distill | — | benchmark |
