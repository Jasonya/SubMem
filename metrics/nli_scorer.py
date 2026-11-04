"""NLI-based faithfulness scorer using cross-encoder/nli-deberta-v3-base.

References:
- TRUE benchmark (Honovich et al., ACL 2022)
- RAGAS (Es et al., EACL 2024)
- FActScore (Min et al., EMNLP 2023)

Pipeline:
  1. Decompose prediction into atomic claims (LLM or rule-based)
  2. For each claim, run NLI against retrieved context (premise=context, hypothesis=claim)
  3. P5 = fraction of claims classified as "entailment"
"""
import re
from typing import Dict, List, Optional, Tuple

import torch

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# NLI label mapping for cross-encoder/nli-deberta-v3-base
# Output logits order: [contradiction, neutral, entailment]
_LABEL_CONTRADICTION = 0
_LABEL_NEUTRAL = 1
_LABEL_ENTAILMENT = 2


class NLIScorer:
    """Loads DeBERTa-v3-NLI and scores (premise, hypothesis) pairs."""

    def __init__(self, model_name: str = _NLI_MODEL_NAME, device: str = None):
        self._model = None
        self._tokenizer = None
        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model = self._model.to(self._device).eval()

    @torch.no_grad()
    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Run NLI on a single (premise, hypothesis) pair.

        Returns dict with scores for each label and the predicted label.
        """
        inputs = self._tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512,
        ).to(self._device)
        logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0).cpu().tolist()
        pred_label = logits.argmax().item()
        return {
            "contradiction": probs[_LABEL_CONTRADICTION],
            "neutral": probs[_LABEL_NEUTRAL],
            "entailment": probs[_LABEL_ENTAILMENT],
            "label": ["contradiction", "neutral", "entailment"][pred_label],
        }

    @torch.no_grad()
    def predict_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[Dict[str, float]]:
        """Run NLI on a batch of (premise, hypothesis) pairs."""
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            inputs = self._tokenizer(
                premises, hypotheses,
                return_tensors="pt", truncation=True, max_length=512, padding=True,
            ).to(self._device)
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            pred_labels = logits.argmax(dim=1).cpu().tolist()
            for prob, pred in zip(probs, pred_labels):
                results.append({
                    "contradiction": prob[_LABEL_CONTRADICTION],
                    "neutral": prob[_LABEL_NEUTRAL],
                    "entailment": prob[_LABEL_ENTAILMENT],
                    "label": ["contradiction", "neutral", "entailment"][pred],
                })
        return results

    def get_gpu_memory_bytes(self) -> int:
        if self._model is None or self._device == "cpu":
            return 0
        return sum(p.numel() * p.element_size() for p in self._model.parameters())


# ── Claim decomposition ──────────────────────────────────────

DECOMPOSE_PROMPT = """Break down the following text into independent atomic facts.
Each fact should be a single, self-contained statement.
One fact per line, no numbering, no bullets.
If the text is already a single fact, just return it as-is.

Text: {text}

Atomic facts:"""


def decompose_claims_llm(text: str, model) -> List[str]:
    """Use an LLM to decompose text into atomic claims (FActScore-style).

    Args:
        text: the prediction text to decompose
        model: a model with .generate(prompt, max_new_tokens) method
    """
    if not text or not text.strip():
        return []
    prompt = DECOMPOSE_PROMPT.format(text=text.strip())
    response = model.generate(prompt, max_new_tokens=128)
    claims = _parse_claims(response)
    # Fallback: if LLM returned nothing useful, use the original text
    if not claims:
        claims = _decompose_claims_rule(text)
    return claims


def decompose_claims_rule(text: str) -> List[str]:
    """Rule-based claim decomposition (sentence splitting + conjunction splitting)."""
    return _decompose_claims_rule(text)


def _decompose_claims_rule(text: str) -> List[str]:
    """Split text into atomic-ish claims using rules."""
    if not text or not text.strip():
        return []
    text = text.strip()

    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    claims = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 3:
            continue
        # Split "A and B" / "A, B, and C" style conjunctions
        # Only split on " and " if both sides are substantial (>3 words each)
        if " and " in sent:
            parts = sent.split(" and ", 1)
            if len(parts[0].split()) >= 3 and len(parts[1].split()) >= 3:
                claims.extend([parts[0].strip().rstrip(","), parts[1].strip()])
                continue
        claims.append(sent)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in claims:
        c_lower = c.lower().strip()
        if c_lower not in seen and len(c_lower) > 2:
            seen.add(c_lower)
            unique.append(c)
    return unique if unique else [text]


def _parse_claims(response: str) -> List[str]:
    """Parse LLM response into list of claims."""
    claims = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or len(line) <= 3:
            continue
        # Strip list prefixes
        for prefix in ["- ", "* ", "• "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        # Strip numbering: "1. ", "1) ", "01. "
        line = re.sub(r'^\d{1,2}[.)]\s*', '', line)
        line = line.strip()
        if line and len(line) > 3:
            claims.append(line)
    return claims


# ── P5 Faithfulness computation ────────────────────────────────

def compute_faithfulness_nli_score(
    claims: List[str],
    sources: List[str],
    nli_scorer: "NLIScorer",
) -> Dict[str, any]:
    """Compute P5 faithfulness for a single prediction.

    Args:
        claims: atomic claims decomposed from prediction
        sources: retrieved memory entries (context)
        nli_scorer: loaded NLIScorer instance

    Returns:
        Dict with P5 score and per-claim details.
    """
    if not claims:
        return {"p5_score": 0.0, "num_claims": 0, "claim_details": []}
    if not sources:
        return {"p5_score": 0.0, "num_claims": len(claims), "claim_details": [
            {"claim": c, "label": "neutral", "entailment_prob": 0.0} for c in claims
        ]}

    # Concatenate all retrieved sources as the premise
    premise = " ".join(sources)

    # Score each claim against the full context
    pairs = [(premise, claim) for claim in claims]
    nli_results = nli_scorer.predict_batch(pairs)

    entailed = 0
    claim_details = []
    for claim, nli_res in zip(claims, nli_results):
        is_entailed = nli_res["label"] == "entailment"
        if is_entailed:
            entailed += 1
        claim_details.append({
            "claim": claim,
            "label": nli_res["label"],
            "entailment_prob": round(nli_res["entailment"], 4),
            "contradiction_prob": round(nli_res["contradiction"], 4),
        })

    p4 = entailed / len(claims)
    return {
        "p5_score": round(p4, 4),
        "num_claims": len(claims),
        "num_entailed": entailed,
        "claim_details": claim_details,
    }
