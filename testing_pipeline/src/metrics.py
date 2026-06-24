"""Scoring functions for evaluating model predictions against references.

Metric types supported:
  - mcq_generative    – extract letter choice from text and compare
  - exact_numeric     – extract number (#### pattern or last number) and compare
  - token_f1          – SQuAD-style token-level F1
  - classification    – discrete numeric label comparison
  - exact_match_normalized – normalized text exact match
"""

import re
from typing import Any, List
from collections import Counter


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_mcq_choice(text: str) -> str:
    """Extracts a single letter (A, B, C, D, etc) from generation."""
    upper = text.upper()
    patterns = [
        r"\(([A-E])\)",
        r"\bOPTION\s+([A-E])\b",
        r"\bANSWER\s*[:\-]?\s*([A-E])\b",
        r"\b([A-E])\b",
    ]
    for pat in patterns:
        m = re.search(pat, upper)
        if m:
            return m.group(1)

    m = re.search(r"([A-E])", upper)
    return m.group(1) if m else ""


def _extract_number(text: str) -> str:
    """Extracts a number after #### for GSM8K, or the last number in text."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


def _normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, articles, extra whitespace."""
    import string
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def _normalize_squad_text(text: str) -> str:
    """SQuAD-style normalization used for token-level F1/EM."""
    return _normalize_text(str(text or ""))


# ---------------------------------------------------------------------------
# Token-level F1 (SQuAD style)
# ---------------------------------------------------------------------------

def _token_f1_single(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_squad_text(prediction).split()
    ref_tokens = _normalize_squad_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _extract_text_references(target: Any) -> List[str]:
    """Extract references from SQuAD/QA-style targets."""
    if target is None:
        return []

    if isinstance(target, dict):
        if "text" in target and isinstance(target["text"], list):
            return [str(t) for t in target["text"]]
        if "answers" in target and isinstance(target["answers"], list):
            return [str(t) for t in target["answers"]]
        return [str(target)]

    if isinstance(target, list):
        return [str(t) for t in target]

    return [str(target)]


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def score_token_f1(pred: str, target: Any) -> float:
    """Best-reference token-level F1 (SQuAD style)."""
    refs = _extract_text_references(target)
    if not refs:
        return 1.0 if _normalize_squad_text(pred) == "" else 0.0
    return max(_token_f1_single(pred, r) for r in refs)


def score_mcq_generative(pred: str, target: str) -> float:
    predicted_choice = _extract_mcq_choice(pred)
    target_choice = _extract_mcq_choice(target) if target else ""
    return 1.0 if predicted_choice == target_choice else 0.0


def score_exact_numeric(pred: str, target: str) -> float:
    predicted_num = _extract_number(pred)
    target_num = _extract_number(str(target)) if target else ""
    try:
        if predicted_num and target_num and abs(float(predicted_num) - float(target_num)) < 1e-6:
            return 1.0
    except ValueError:
        pass
    return 0.0


def _extract_classification_label(text: str) -> str:
    """Extracts a class label with preference for explicit final-answer patterns."""
    raw = str(text)
    lower = raw.lower()

    prioritized_patterns = [
        r"final\s+answer\s*[:\-]?\s*([0-9]+)",
        r"answer\s*[:\-]?\s*([0-9]+)",
        r"label\s*[:\-]?\s*([0-9]+)",
    ]
    for pat in prioritized_patterns:
        match = re.search(pat, lower)
        if match:
            return match.group(1)

    numbers = re.findall(r"\b([0-9]+)\b", raw)
    return numbers[-1] if numbers else ""


def score_classification(pred: str, target: str) -> float:
    """Compare discrete numeric labels."""
    predicted_label = _extract_classification_label(pred)
    target_label = str(target).strip() if target else ""
    return 1.0 if predicted_label == target_label else 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def score_single(metric_type: str, pred: str, target: Any) -> float:
    """Score a single prediction against a reference using the given metric."""
    if metric_type == "mcq_generative":
        return score_mcq_generative(pred, str(target))
    elif metric_type == "exact_numeric":
        return score_exact_numeric(pred, str(target))
    elif metric_type == "token_f1":
        return score_token_f1(pred, target)
    elif metric_type == "classification":
        return score_classification(pred, str(target))
    elif metric_type == "exact_match_normalized":
        return 1.0 if _normalize_text(pred) == _normalize_text(str(target)) else 0.0
    else:
        # Default: normalized exact match
        return 1.0 if _normalize_text(pred) == _normalize_text(str(target)) else 0.0
