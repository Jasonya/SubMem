"""C6 — Temporal Knowledge Graph Memory (Zep/Graphiti-style).

Reference: Zep / Graphiti (Raghavan et al., 2025)
- "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"
- https://arxiv.org/abs/2501.13956

Key adaptations from the paper:
- Temporal context graph: entity nodes + fact edges with valid_at/invalid_at
- Episode provenance: raw data stored as immutable episodes
- Bi-temporal conflict resolution: new facts invalidate contradicting old facts
- Hybrid search: BM25 + cosine similarity on fact embeddings → RRF reranking
- Entity resolution: LLM-based deduplication of extracted entities

Pipeline: buffer utterances → episode creation → LLM entity extraction →
          entity resolution → LLM fact/edge extraction with temporal bounds →
          temporal conflict resolution → networkx graph

Does NOT inherit DenseRetrievalMemory. Uses own encoder for entity/fact embeddings.
Retrieval: hybrid BM25 + cosine over fact edges → RRF reranking.
"""
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit
# No chunking — Graphiti processes per-episode (per-message)

_DEFAULT_CHUNK_SIZE = 256
# embedding dim is dynamic via self._enc.dim
_RRF_K = 60  # Reciprocal Rank Fusion constant


# ── Data model (faithful to Graphiti) ──────────────────────────

@dataclass
class EntityNode:
    """Graph entity node with embedding for matching."""
    uuid: str
    name: str
    entity_type: str = "Entity"
    summary: str = ""
    name_embedding: Optional[np.ndarray] = None
    created_at: Optional[str] = None


@dataclass
class EpisodicNode:
    """Immutable raw data snapshot (provenance)."""
    uuid: str
    content: str
    source_description: str = ""
    valid_at: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class EntityEdge:
    """Fact/relationship edge with temporal validity."""
    uuid: str
    source_node_uuid: str
    target_node_uuid: str
    relation_type: str
    fact: str  # natural language fact
    fact_embedding: Optional[np.ndarray] = None
    episode_uuid: str = ""
    valid_at: Optional[str] = None   # when fact became true
    invalid_at: Optional[str] = None  # when fact stopped being true
    created_at: Optional[str] = None

    dia_ids: List[str] = field(default_factory=list)  # for P4/P5 evidence matching

    @property
    def is_valid(self) -> bool:
        return self.invalid_at is None


# ── LLM prompts (faithful to Graphiti github.com/getzep/graphiti) ──

# Entity extraction: Graphiti extract_nodes.extract_message style
_ENTITY_PROMPT = """You are an expert entity extractor. Extract all significant entities from the current message.
Use previous messages ONLY for context (resolving pronouns, references), but extract entities ONLY from the current message.

Rules:
- ALWAYS extract the speaker/actor as the first entity
- Use FULL QUALIFIED names when possible: "Nisha's dad" not just "dad", "road cycling" not just "cycling"
- EXCLUDE only: pronouns (he/she/it/they) and very generic words (thing/stuff)
- INCLUDE: people (by name or qualified relation), places, organizations, activities, events, topics discussed
- Entity types: Person, Location, Organization, Event, DateTime, Concept, Object, Other

Output format — one entity per line:
ENTITY_NAME | ENTITY_TYPE

Previous messages (for context only):
{previous_context}

Current message:
{chunk}

Entities:"""

# Entity resolution: Graphiti dedupe_nodes style
_ENTITY_RESOLVE_PROMPT = """Determine if the NEW ENTITY is a duplicate of any EXISTING ENTITY.
Entities are duplicates ONLY if they refer to the same real-world object or concept.

Duplicates include:
- Same entity in different contexts (e.g., "Sam" appearing multiple times)
- Abbreviations or aliases (e.g., "NYC" = "New York City")
- Synonymous references with same possessor (e.g., "car" = "vehicle" for same owner)

NOT duplicates:
- Related but distinct entities
- Same name but different referent (e.g., "Java" language vs "Java" island)

New entity: {new_entity} ({new_type})

Existing entities:
{existing_str}

If duplicate, respond: DUPLICATE: <existing_entity_name>
If not duplicate, respond: NEW

Your answer:"""

_QUERY_ENTITY_PROMPT = """Extract the key entities mentioned in the following question.
Output one entity name per line, no numbering, no explanation.
If no entities found, output NONE.

Question: {query}

Entities:"""

# Edge extraction: Graphiti extract_edges.edge style
_EDGE_PROMPT = """You are an expert fact extractor. Extract factual relationships from the current message.
Use previous messages ONLY for context (resolving pronouns, references). Extract facts ONLY from the current message.

Rules:
- Each fact must involve exactly TWO DISTINCT entities from the entity list
- Replace pronouns with actual entity names
- RELATION_TYPE: short label in SCREAMING_SNAKE_CASE (e.g., LIVES_IN, WORKS_AT, HAPPENED_ON)
- FACT: detailed natural language statement of the relationship
- Temporal: use ISO 8601 dates when mentioned; "ongoing" if currently true; "none" if unknown
- For single-entity states, reframe to involve two entities (e.g., "Alice is happy" → "Alice | HAS_STATE | happiness")

Output format — one fact per line:
SOURCE_ENTITY | RELATION_TYPE | TARGET_ENTITY | FACT | VALID_AT | INVALID_AT

Reference time: {reference_time}

Previous messages (for context only):
{previous_context}

Current message:
{chunk}

Entities:
{entities_str}

Facts:"""

# Edge conflict resolution: Graphiti dedupe_edges style
_EDGE_RESOLVE_PROMPT = """You are a fact-checking assistant. Compare a NEW FACT against EXISTING FACTS.

Determine:
1. Is the new fact a DUPLICATE? (identical factual information — same entities, same relationship, same details)
2. Does the new fact CONTRADICT any existing fact? (conflicts semantically — e.g., different role, different location, different status)

IMPORTANT:
- NEVER mark facts with key differences in numbers, dates, or qualifiers as duplicates
- Different events on different days are neither duplicates nor contradictions
- A fact CAN be both a duplicate AND contradict other facts simultaneously

Existing facts:
{existing_str}

New fact: {new_fact}

Respond with ONE line:
- DUPLICATE: <existing_idx> (if identical information)
- CONTRADICTS: <existing_idx> (if new fact invalidates existing — the old fact should be marked as no longer valid)
- NEW (if genuinely new information, not present in and not conflicting with existing facts)

Your answer:"""


def _mean_pool(model_output, attention_mask):
    token_embs = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    summed = torch.sum(token_embs * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


@register_memory_unit("C6")
class KnowledgeGraphMemory(BaseMemoryUnit):
    """Temporal Knowledge Graph Memory — Zep/Graphiti-style.

    Does NOT inherit DenseRetrievalMemory. Faithful to Graphiti's pipeline:
    - Write: episode → entity extraction → entity resolution → fact extraction
             with temporal bounds → temporal conflict resolution → graph
    - Read:  hybrid search (BM25 + cosine on fact edges) → RRF reranking
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        import networkx as nx
        self._graph = nx.MultiDiGraph()
        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict] = []
        self._indexed = False

        # Core data stores
        self._entity_nodes: Dict[str, EntityNode] = {}  # uuid → node
        self._entity_edges: List[EntityEdge] = []        # all fact edges
        self._episodes: List[EpisodicNode] = []          # provenance
        self._entity_name_map: Dict[str, str] = {}       # name → uuid

        # Stats tracking
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._op_counts = {"ADD": 0, "DUPLICATE": 0, "CONTRADICT": 0}
        self._llm_call_times: List[float] = []
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }
        self._chunk_size = (
            config.extra.get("chunk_size", _DEFAULT_CHUNK_SIZE)
            if hasattr(config, "extra") and config.extra
            else _DEFAULT_CHUNK_SIZE
        )
        self._entity_match_threshold = 0.7
        self._uuid_counter = 0

        # Encoder for entity/fact embeddings
        self._encoder = None
        self._enc_tokenizer = None
        self._enc_device = "cpu"
        self._init_encoder()

    def _next_uuid(self) -> str:
        self._uuid_counter += 1
        return f"n{self._uuid_counter:06d}"

    def _init_encoder(self):
        from .embeddings import create_encoder
        self._enc = create_encoder(self.config)

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self._enc.encode(texts)

    def _track_llm_call(self, prompt: str, response: str):
        self._token_counts["write_llm_input"] += len(self.model.encode(prompt))
        self._token_counts["write_llm_output"] += len(self.model.encode(response))
        self._token_counts["write_llm_calls"] += 1

    def _track_mgmt_call(self, prompt: str, response: str):
        self._token_counts.setdefault("mgmt_llm_input", 0)
        self._token_counts.setdefault("mgmt_llm_output", 0)
        self._token_counts.setdefault("mgmt_llm_calls", 0)
        self._token_counts["mgmt_llm_input"] += len(self.model.encode(prompt))
        self._token_counts["mgmt_llm_output"] += len(self.model.encode(response))
        self._token_counts["mgmt_llm_calls"] += 1

    # ── Write ──────────────────────────────────────────────────

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._indexed = False
        self._write_call_count += 1

    # ── Step 1: Entity extraction ──────────────────────────────

    def _extract_entities(self, chunk: str, previous_context: str = "") -> List[Tuple[str, str]]:
        """LLM entity extraction (Graphiti extract_nodes)."""
        prompt = _ENTITY_PROMPT.format(
            chunk=chunk,
            previous_context=previous_context or "(no previous messages)",
        )
        t0 = time.perf_counter()
        response = self.model.generate(prompt, max_new_tokens=256)
        self._llm_call_times.append(time.perf_counter() - t0)
        self._track_llm_call(prompt, response)

        entities = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2 and parts[0] and parts[1]:
                name = parts[0].lower().replace(" ", "_")
                etype = parts[1].strip()
                entities.append((name, etype))
        return entities

    # ── Step 2: Entity resolution (Graphiti dedupe_nodes) ──────

    def _resolve_entity(self, name: str, etype: str) -> str:
        """Resolve entity against existing graph. Returns uuid."""
        # Exact name match
        if name in self._entity_name_map:
            return self._entity_name_map[name]

        # Embedding similarity match
        if self._entity_nodes:
            new_emb = self._encode([name.replace("_", " ")])[0]
            best_match = None
            best_sim = 0.0
            for existing_name, uuid in self._entity_name_map.items():
                node = self._entity_nodes[uuid]
                if node.name_embedding is not None:
                    sim = float(np.dot(new_emb, node.name_embedding))
                    if sim > best_sim:
                        best_sim = sim
                        best_match = (existing_name, uuid)

            # If close match, ask LLM to confirm (Graphiti-style)
            if best_match and best_sim > self._entity_match_threshold:
                existing_str = "\n".join(
                    f"- {n.name} ({n.entity_type})"
                    for n in self._entity_nodes.values()
                    if n.name_embedding is not None
                    and float(np.dot(new_emb, n.name_embedding)) > self._entity_match_threshold
                )
                prompt = _ENTITY_RESOLVE_PROMPT.format(
                    new_entity=name.replace("_", " "),
                    new_type=etype,
                    existing_str=existing_str,
                )
                t0 = time.perf_counter()
                response = self.model.generate(prompt, max_new_tokens=32)
                self._llm_call_times.append(time.perf_counter() - t0)
                self._track_mgmt_call(prompt, response)

                first_line = response.strip().split("\n")[0].upper()
                if "DUPLICATE:" in first_line:
                    dup_name = response.strip().split(":")[-1].strip().lower().replace(" ", "_")
                    if dup_name in self._entity_name_map:
                        return self._entity_name_map[dup_name]

        # New entity
        uuid = self._next_uuid()
        emb = self._encode([name.replace("_", " ")])[0]
        node = EntityNode(
            uuid=uuid,
            name=name,
            entity_type=etype,
            name_embedding=emb,
            created_at=datetime.now().isoformat(),
        )
        self._entity_nodes[uuid] = node
        self._entity_name_map[name] = uuid
        self._graph.add_node(uuid, name=name, type=etype)
        return uuid

    # ── Step 3: Fact/Edge extraction with temporal bounds ──────

    def _extract_edges(self, chunk: str, entities: List[Tuple[str, str]],
                       reference_time: str = "", previous_context: str = "") -> List[Dict]:
        """LLM fact extraction (Graphiti extract_edges) with temporal bounds."""
        if not entities:
            return []
        entities_str = "\n".join(f"{name} ({etype})" for name, etype in entities)
        if not reference_time:
            reference_time = datetime.now().isoformat()

        prompt = _EDGE_PROMPT.format(
            chunk=chunk,
            entities_str=entities_str,
            reference_time=reference_time,
            previous_context=previous_context or "(no previous messages)",
        )
        t0 = time.perf_counter()
        response = self.model.generate(prompt, max_new_tokens=512)
        self._llm_call_times.append(time.perf_counter() - t0)
        self._track_llm_call(prompt, response)

        edges = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                src = parts[0].lower().replace(" ", "_")
                rel = parts[1].upper().replace(" ", "_")
                dst = parts[2].lower().replace(" ", "_")
                fact = parts[3]
                valid_at = parts[4] if len(parts) > 4 and parts[4].lower() != "ongoing" else None
                invalid_at = parts[5] if len(parts) > 5 and parts[5].lower() != "none" else None
                if src and rel and dst and fact:
                    edges.append({
                        "src": src, "rel": rel, "dst": dst,
                        "fact": fact,
                        "valid_at": valid_at,
                        "invalid_at": invalid_at,
                    })
        return edges

    # ── Step 4: Temporal conflict resolution (Graphiti-style) ──

    def _get_existing_edges_between(self, src_uuid: str, dst_uuid: str) -> List[EntityEdge]:
        """Get existing valid edges between two entity nodes."""
        return [
            e for e in self._entity_edges
            if e.source_node_uuid == src_uuid
            and e.target_node_uuid == dst_uuid
            and e.is_valid
        ]

    def _resolve_edge(self, new_edge_data: Dict, src_uuid: str, dst_uuid: str,
                      episode_uuid: str) -> Optional[EntityEdge]:
        """Resolve a new edge against existing edges.

        Graphiti-style: detect duplicates and contradictions,
        invalidate contradicted facts with temporal bounds.
        """
        existing = self._get_existing_edges_between(src_uuid, dst_uuid)
        # Also check reverse direction
        existing += self._get_existing_edges_between(dst_uuid, src_uuid)

        if not existing:
            self._op_counts["ADD"] += 1
            return self._create_edge(new_edge_data, src_uuid, dst_uuid, episode_uuid)

        # LLM-based resolution
        existing_str = "\n".join(
            f"[{i}] {e.fact} (valid_at: {e.valid_at or 'unknown'}, invalid_at: {e.invalid_at or 'none'})"
            for i, e in enumerate(existing)
        )
        prompt = _EDGE_RESOLVE_PROMPT.format(
            existing_str=existing_str,
            new_fact=new_edge_data["fact"],
        )
        t0 = time.perf_counter()
        response = self.model.generate(prompt, max_new_tokens=32)
        self._llm_call_times.append(time.perf_counter() - t0)
        self._track_mgmt_call(prompt, response)

        first_line = response.strip().upper().split("\n")[0]

        if "DUPLICATE" in first_line:
            self._op_counts["DUPLICATE"] += 1
            return None

        if "CONTRADICT" in first_line:
            self._op_counts["CONTRADICT"] += 1
            # Extract index of contradicted fact
            try:
                idx = int(re.search(r'\d+', first_line.split("CONTRADICT")[-1]).group())
                if 0 <= idx < len(existing):
                    # Invalidate old fact (Graphiti temporal invalidation)
                    old_edge = existing[idx]
                    old_edge.invalid_at = new_edge_data.get("valid_at") or datetime.now().isoformat()
            except (ValueError, AttributeError, IndexError):
                pass
            return self._create_edge(new_edge_data, src_uuid, dst_uuid, episode_uuid)

        # NEW
        self._op_counts["ADD"] += 1
        return self._create_edge(new_edge_data, src_uuid, dst_uuid, episode_uuid)

    def _create_edge(self, edge_data: Dict, src_uuid: str, dst_uuid: str,
                     episode_uuid: str) -> EntityEdge:
        """Create and store a new EntityEdge."""
        edge = EntityEdge(
            uuid=self._next_uuid(),
            source_node_uuid=src_uuid,
            target_node_uuid=dst_uuid,
            relation_type=edge_data["rel"],
            fact=edge_data["fact"],
            fact_embedding=self._encode([edge_data["fact"]])[0],
            episode_uuid=episode_uuid,
            valid_at=edge_data.get("valid_at"),
            invalid_at=edge_data.get("invalid_at"),
            created_at=datetime.now().isoformat(),
        )
        self._entity_edges.append(edge)
        self._graph.add_edge(
            src_uuid, dst_uuid,
            key=edge.uuid,
            relation=edge.relation_type,
            fact=edge.fact,
            valid_at=edge.valid_at,
            invalid_at=edge.invalid_at,
        )
        self._stored_token_count += len(self.model.encode(edge.fact))
        return edge

    # ── Indexing pipeline ──────────────────────────────────────

    def _ensure_indexed(self):
        if self._indexed:
            return
        if not self._buffer:
            self._indexed = True
            return

        t0 = time.perf_counter()
        self._raw_token_count = sum(len(self.model.encode(t)) for t in self._buffer)

        n_msgs = len(self._buffer)
        n_prev = 4  # Graphiti uses last 4 messages for context
        print(f"  C6: building temporal KG from {n_msgs} messages (per-episode, Graphiti-style)...", flush=True)
        for ci, text in enumerate(self._buffer):
            meta = self._buffer_metadata[ci] if ci < len(self._buffer_metadata) else {}
            dia_id = meta.get("dia_id", "")
            reference_time = meta.get("date", datetime.now().isoformat())

            # Build previous message context (Graphiti: last 4 messages)
            prev_start = max(0, ci - n_prev)
            previous_context = "\n".join(self._buffer[prev_start:ci]) if ci > 0 else ""

            # Step 0: Create episode (Graphiti provenance)
            episode = EpisodicNode(
                uuid=self._next_uuid(),
                content=text,
                source_description=meta.get("session_id", ""),
                valid_at=reference_time,
                created_at=datetime.now().isoformat(),
            )
            self._episodes.append(episode)

            # Step 1: Entity extraction (with previous context)
            entities = self._extract_entities(text, previous_context)

            # Step 2: Entity resolution
            resolved = {}
            for name, etype in entities:
                uuid = self._resolve_entity(name, etype)
                resolved[name] = uuid

            # Step 3: Fact extraction with temporal bounds (with previous context)
            raw_edges = self._extract_edges(text, entities, reference_time, previous_context)

            if (ci + 1) % 20 == 0 or ci + 1 == n_msgs:
                print(f"    [{ci+1}/{n_msgs}] {len(self._entity_nodes)} nodes, "
                      f"{len(self._entity_edges)} edges", flush=True)

            # Step 4: Temporal conflict resolution
            for edge_data in raw_edges:
                src_name = edge_data["src"]
                dst_name = edge_data["dst"]
                src_uuid = resolved.get(src_name)
                dst_uuid = resolved.get(dst_name)
                if not src_uuid or not dst_uuid:
                    continue
                edge = self._resolve_edge(edge_data, src_uuid, dst_uuid, episode.uuid)
                if edge is not None and dia_id:
                    edge.dia_ids = [dia_id]

        # Amortize indexing time across all edges
        total_time = time.perf_counter() - t0
        n_edges = len(self._entity_edges) or 1
        per_entry = total_time / n_edges
        self._write_times.extend([per_entry] * n_edges)

        self._indexed = True

    # ── Read (Graphiti-style: BM25 + cosine → RRF) ────────────

    @staticmethod
    def _bm25_score(query_tokens: List[str], doc_tokens: List[str],
                    avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
        if not doc_tokens or not query_tokens:
            return 0.0
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            if qt in tf:
                f = tf[qt]
                numerator = f * (k1 + 1)
                denominator = f + k1 * (1 - b + b * dl / max(avg_dl, 1))
                score += numerator / denominator
        return score

    @staticmethod
    def _rrf_combine(ranked_lists: List[List[Tuple[int, float]]],
                     k: int = _RRF_K) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion (Graphiti default reranker)."""
        scores: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (idx, _) in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
        combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return combined

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        self._ensure_indexed()
        valid_edges = [e for e in self._entity_edges if e.is_valid]
        if not valid_edges:
            return []

        t0 = time.perf_counter()

        # ── Search 1: BM25 over fact text ──
        query_tokens = query.lower().split()
        avg_dl = sum(len(e.fact.lower().split()) for e in valid_edges) / max(len(valid_edges), 1)

        bm25_ranked = []
        for i, edge in enumerate(valid_edges):
            doc_tokens = edge.fact.lower().split()
            score = self._bm25_score(query_tokens, doc_tokens, avg_dl)
            bm25_ranked.append((i, score))
        bm25_ranked.sort(key=lambda x: x[1], reverse=True)

        # ── Search 2: Cosine similarity over fact embeddings ──
        q_emb = self._encode([query])[0]
        cosine_ranked = []
        for i, edge in enumerate(valid_edges):
            if edge.fact_embedding is not None:
                sim = float(np.dot(q_emb, edge.fact_embedding))
                cosine_ranked.append((i, sim))
        cosine_ranked.sort(key=lambda x: x[1], reverse=True)

        # ── Search 3: BFS graph traversal (Graphiti-style) ──
        # LLM extracts entities from query → match to graph nodes → 1-hop edges
        bfs_ranked = []
        prompt = _QUERY_ENTITY_PROMPT.format(query=query)
        entity_response = self.model.generate(prompt, max_new_tokens=64)

        # Track retrieval LLM cost (E8/E9)
        self._token_counts.setdefault("read_llm_input", 0)
        self._token_counts.setdefault("read_llm_output", 0)
        self._token_counts.setdefault("read_llm_calls", 0)
        self._token_counts["read_llm_input"] += len(self.model.encode(prompt))
        self._token_counts["read_llm_output"] += len(self.model.encode(entity_response))
        self._token_counts["read_llm_calls"] += 1

        # Parse extracted entities and match to graph nodes
        query_entities = []
        if "NONE" not in entity_response.strip().upper():
            for line in entity_response.strip().split("\n"):
                ent = line.strip().lower().replace(" ", "_")
                if ent and len(ent) > 1:
                    query_entities.append(ent)

        matched_nodes = []
        for ent in query_entities:
            for uuid, node in self._entity_nodes.items():
                if ent in node.name or node.name in ent:
                    matched_nodes.append(uuid)
        if matched_nodes:
            bfs_edge_indices = set()
            for node_uuid in matched_nodes:
                for i, edge in enumerate(valid_edges):
                    if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                        bfs_edge_indices.add(i)
            # Rank BFS results by cosine similarity to query
            for i in bfs_edge_indices:
                edge = valid_edges[i]
                sim = float(np.dot(q_emb, edge.fact_embedding)) if edge.fact_embedding is not None else 0.0
                bfs_ranked.append((i, sim))
            bfs_ranked.sort(key=lambda x: x[1], reverse=True)

        # ── RRF reranking (Graphiti default) ──
        ranked_lists = [bm25_ranked, cosine_ranked]
        if bfs_ranked:
            ranked_lists.append(bfs_ranked)
        combined = self._rrf_combine(ranked_lists)

        # Build results — include temporal metadata in content
        # Graphiti returns raw edge objects with fact + valid_at + invalid_at;
        # callers format as needed. We embed temporal info in content so the
        # generation model can reason about time.
        results = []
        for idx, rrf_score in combined[:top_k]:
            edge = valid_edges[idx]
            src_name = self._entity_nodes.get(edge.source_node_uuid, EntityNode("", "?")).name
            dst_name = self._entity_nodes.get(edge.target_node_uuid, EntityNode("", "?")).name
            content = f"[{src_name} -> {dst_name}] {edge.fact}"
            if edge.valid_at:
                content += f" (valid from: {edge.valid_at})"
            if edge.invalid_at:
                content += f" (valid until: {edge.invalid_at})"
            results.append(MemoryEntry(
                content=content,
                metadata={
                    "src": src_name,
                    "rel": edge.relation_type,
                    "dst": dst_name,
                    "valid_at": edge.valid_at,
                    "dia_ids": edge.dia_ids,
                },
                score=rrf_score,
            ))

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            len(self.model.encode(r.content)) for r in results
        )
        return results

    # ── Lifecycle ──────────────────────────────────────────────

    def reset(self):
        import networkx as nx
        self._graph = nx.MultiDiGraph()
        self._entity_nodes.clear()
        self._entity_edges.clear()
        self._episodes.clear()
        self._entity_name_map.clear()
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._indexed = False
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._op_counts = {"ADD": 0, "DUPLICATE": 0, "CONTRADICT": 0}
        self._llm_call_times.clear()
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._uuid_counter = 0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }

    def size(self) -> int:
        return sum(1 for e in self._entity_edges if e.is_valid)

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_indexed()
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        write_times_ms = [t * 1000 for t in self._write_times]
        cr = (self._raw_token_count / self._stored_token_count
              if self._stored_token_count > 0 else 1.0)

        num_entities = len(self._entity_nodes)
        num_valid_edges = sum(1 for e in self._entity_edges if e.is_valid)
        num_invalid_edges = sum(1 for e in self._entity_edges if not e.is_valid)

        stats.update({
            "compression_ratio": round(cr, 2),
            "raw_token_count": self._raw_token_count,
            "stored_token_count": self._stored_token_count,
            "num_entities": num_entities,
            "num_valid_edges": num_valid_edges,
            "num_invalid_edges": num_invalid_edges,
            "num_episodes": len(self._episodes),
            "graph_density": (
                num_valid_edges / (num_entities * (num_entities - 1))
                if num_entities > 1 else 0.0
            ),
            "conflict_ops": dict(self._op_counts),
            "num_raw_utterances": len(self._buffer),
            "avg_llm_call_ms": (
                sum(self._llm_call_times) / len(self._llm_call_times) * 1000
                if self._llm_call_times else 0.0
            ),
            "memory_size_bytes": self.get_external_storage_bytes(),
            "token_consumption": dict(self._token_counts),
            "write_call_count": self._write_call_count,
            "read_latency_ms": {
                "mean": float(np.mean(read_times_ms)) if read_times_ms else 0.0,
                "p50": float(np.median(read_times_ms)) if read_times_ms else 0.0,
                "p90": float(np.percentile(read_times_ms, 90)) if read_times_ms else 0.0,
            },
            "write_latency_ms": {
                "mean": float(np.mean(write_times_ms)) if write_times_ms else 0.0,
                "p50": float(np.median(write_times_ms)) if write_times_ms else 0.0,
                "p90": float(np.percentile(write_times_ms, 90)) if write_times_ms else 0.0,
            },
        })
        return stats

    def get_external_storage_bytes(self) -> int:
        # Entity nodes (name + embedding)
        node_bytes = sum(
            len(n.name.encode("utf-8")) + (self._enc.dim * 4 if n.name_embedding is not None else 0)
            for n in self._entity_nodes.values()
        )
        # Fact edges (fact text + embedding)
        edge_bytes = sum(
            len(e.fact.encode("utf-8")) + (self._enc.dim * 4 if e.fact_embedding is not None else 0)
            for e in self._entity_edges
        )
        # Episodes are provenance/audit log, not working memory — excluded from E1
        return node_bytes + edge_bytes

    def get_all_entries(self) -> List[MemoryEntry]:
        return [
            MemoryEntry(
                content=e.fact,
                metadata={
                    "src": self._entity_nodes.get(e.source_node_uuid, EntityNode("", "?")).name,
                    "rel": e.relation_type,
                    "dst": self._entity_nodes.get(e.target_node_uuid, EntityNode("", "?")).name,
                    "valid_at": e.valid_at,
                    "invalid_at": e.invalid_at,
                    "dia_ids": e.dia_ids,
                },
            )
            for e in self._entity_edges
            if e.is_valid
        ]
