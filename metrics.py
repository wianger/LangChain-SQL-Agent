"""Evaluation metrics for QA: token-level F1, recall, and accuracy (substring)."""

from __future__ import annotations

import collections
import json
import re
import string
from collections import Counter
from typing import Any, Dict, List, Union

from langchain_core.messages import HumanMessage, SystemMessage

REFUSAL_KEYWORDS = [
    "not mentioned",
    "no information",
    "cannot be answered",
    "none",
    "unknown",
    "don't know",
    "unanswerable",
    "No/insufficient information",
]


def _normalize(s: str) -> str:
    """标准化答案文本：去标点、转小写、去冠词"""
    s = str(s).replace(",", "")

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _get_tokens(text: str) -> List[str]:
    return _normalize(text).split()


def _is_refusal(text: str) -> bool:
    return any(r in text.lower() for r in REFUSAL_KEYWORDS)


# ── F1 (single gold answer) ─────────────────────────────────────────────────
def _token_f1_single(prediction: str, ground_truth: str) -> float:
    pred_tokens = _get_tokens(prediction)
    truth_tokens = _get_tokens(ground_truth)
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def token_f1(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """Token-level F1.  If *ground_truth* is a list, return the max F1."""
    if isinstance(ground_truth, list):
        return max(
            (_token_f1_single(prediction, gt) for gt in ground_truth), default=0.0
        )
    return _token_f1_single(prediction, ground_truth)


# ── Recall ───────────────────────────────────────────────────────────────────
def _token_recall_single(
    retrieved_texts: List[str],
    evidence_list: List[str],
    soft_threshold: float = 0.8,
    min_soft_match_tokens: int = 4,
) -> float:
    """Recall combining strict substring match with soft token-overlap match.

    For each evidence:
      1. Strict substring match against concatenated retrieved text.
      2. If the evidence has fewer effective tokens than *min_soft_match_tokens*,
         skip soft matching (prevents short-ID false positives).
      3. Otherwise, compute token coverage; hit if >= *soft_threshold*.
    Final score = hit_count / len(evidence_list).
    """
    if not evidence_list:
        return 0.0

    combined_retrieved = " ".join(retrieved_texts)
    normalized_retrieved = _normalize(combined_retrieved)
    ret_tokens = set(normalized_retrieved.split())

    hit_count = 0

    for evidence in evidence_list:
        # Step 1: strict substring match (unnormalized)
        if evidence in combined_retrieved:
            hit_count += 1
            continue

        # Normalise evidence for token-level comparison
        normalized_ev = _normalize(evidence)
        ev_tokens = set(normalized_ev.split())

        if not ev_tokens:
            continue

        # Length blocking: short evidence must match strictly
        if len(ev_tokens) < min_soft_match_tokens:
            continue

        # Step 2: soft token-overlap match (long evidence only)
        overlap_count = len(ev_tokens & ret_tokens)
        coverage = overlap_count / len(ev_tokens)

        if coverage >= soft_threshold:
            hit_count += 1

    return hit_count / len(evidence_list)


def token_recall(
    prediction: str,
    evidences: Union[str, List[str]],
    soft_threshold: float = 0.8,
    min_soft_match_tokens: int = 4,
    retrieved_texts: List[str] | None = None,
) -> float:
    """Public recall entry-point.

    If *retrieved_texts* is provided (the raw SQL query results from the
    agent's intermediate steps), recall is computed against those.
    Otherwise falls back to using *prediction* (the agent's final answer).
    *evidences* (str or list of str) is the evidence list.
    """
    retrieved = retrieved_texts if retrieved_texts else [prediction]
    evidence = evidences if isinstance(evidences, list) else [evidences]
    return _token_recall_single(
        retrieved, evidence, soft_threshold, min_soft_match_tokens
    )


# ── Accuracy (LLM-as-judge) ──────────────────────────────────────────────────
def llm_grader(
    llm,
    question: str,
    gold_answer: str,
    response: str,
    dataset_name: str = "Locomo",
) -> dict:
    """Use an LLM judge to score a generated answer against a gold answer.

    Returns ``{"score": int, "reasoning": str, "prompt_type": str}``.
    LoCoMo → score 0 or 4;  Generic → score 0-4.
    """
    dataset_name_lower = (dataset_name or "").lower()
    content = ""
    score = 0
    reasoning = "No reasoning provided."
    prompt_type = "Generic_0-4"

    if "locomo" in dataset_name_lower:
        prompt_type = "Locomo_0or4"
        system_prompt = (
            "You are an expert grader that determines if answers to questions "
            "match a gold standard answer"
        )
        accuracy_prompt = f"""
Your task is to label an answer to a question by assigning a score of 4 or 0. You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer

which you will score as 4 or 0.
The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as correct.
For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as correct. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it correct if it's the same date.

Scoring rule:
- Output score 4 if the generated answer should be considered CORRECT.
- Output score 0 if the generated answer should be considered WRONG.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning.
Respond with JSON only: {{"score": 4 or 0, "reasoning": "your explanation"}}
"""
    else:
        prompt_type = "Generic_0-4"
        system_prompt = (
            "You are an expert evaluator scoring how well an AI-generated "
            "answer matches a gold standard (ground truth)."
        )
        accuracy_prompt = f"""
Please score the Generated Answer against the Gold Answer on a scale of 0 to 4.

[Evaluation Rubric]
- Score 4 (Perfect): Completely and accurately captures the core meaning and facts of the gold answer.
- Score 3 (Good): Captures the main facts but includes unnecessary verbosity or minor non-contradictory details.
- Score 2 (Partial): Missing some key factual information but touches on the correct topic.
- Score 1 (Poor): Mostly incorrect or severely incomplete.
- Score 0 (Wrong): Completely wrong, contradicts the gold answer, or hallucinates.

Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {response}

First, write a 1-sentence reasoning. Then output the integer score.
Respond ONLY with a JSON object: {{"score": 0 to 4, "reasoning": "string"}}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=accuracy_prompt),
    ]

    try:
        resp = llm.invoke(messages)
        content = resp.content if resp and hasattr(resp, "content") else ""

        result = json.loads(content)
        score = int(result.get("score", 0))
        reasoning = result.get("reasoning", "No reasoning provided.")

        if "locomo" in dataset_name_lower:
            score = 4 if score == 4 else 0
        else:
            score = max(0, min(4, score))

    except Exception:
        text = (content or "").strip()
        reasoning = (
            f"Parse fallback from raw output: {text}"
            if text
            else "Parse failed or model invocation failed. Defaulted to 0."
        )

        match = re.search(r'"score"\s*:\s*([0-4])', text)
        if match:
            score = int(match.group(1))
        else:
            match = re.search(r"\b([0-4])\b", text)
            if match:
                score = int(match.group(1))
            else:
                score = 0

        if "locomo" in dataset_name_lower:
            score = 4 if score == 4 else 0
        else:
            score = max(0, min(4, score))

    return {
        "score": score,
        "reasoning": reasoning,
        "prompt_type": prompt_type,
    }


def accuracy(
    llm,
    question: str,
    prediction: str,
    ground_truth: Union[str, List[str]],
    dataset_name: str = "Locomo",
) -> dict:
    """LLM-judge accuracy.  Returns the best grader result dict across gold answers."""
    if isinstance(ground_truth, list):
        results = [
            llm_grader(llm, question, prediction, gt, dataset_name)
            for gt in ground_truth
        ]
        return (
            max(results, key=lambda r: r["score"])
            if results
            else {"score": 0, "reasoning": "No ground truth", "prompt_type": "N/A"}
        )
    return llm_grader(llm, question, prediction, ground_truth, dataset_name)


# ── Aggregate ────────────────────────────────────────────────────────────────
def compute_metrics(
    llm,
    questions: List[str],
    predictions: List[str],
    ground_truths: List[Union[str, List[str]]],
    dataset_name: str = "Locomo",
    evidences: List[Union[str, List[str]]] | None = None,
    retrieved_texts_list: List[List[str]] | None = None,
) -> Dict[str, Any]:
    """Return averaged F1, recall, and accuracy over (pred, gt) pairs.

    *evidences* overrides ground_truths as the recall target (what to match).
    *retrieved_texts_list* overrides predictions as the recall source
    (the actual DB query results from agent intermediate steps).
    ``accuracy`` is normalised to 0.0-1.0 (raw score / 4).
    """
    assert len(predictions) == len(ground_truths)
    n = len(predictions)
    if n == 0:
        return {"f1": 0.0, "recall": 0.0, "accuracy": 0.0, "accuracy_raw": 0.0}
    recall_targets = evidences if evidences is not None else ground_truths
    f1s = [token_f1(p, g) for p, g in zip(predictions, ground_truths)]
    if retrieved_texts_list is not None:
        recalls = [
            token_recall(p, e, retrieved_texts=rt)
            for p, e, rt in zip(predictions, recall_targets, retrieved_texts_list)
        ]
    else:
        recalls = [token_recall(p, e) for p, e in zip(predictions, recall_targets)]
    acc_results = [
        accuracy(llm, q, p, g, dataset_name)
        for q, p, g in zip(questions, predictions, ground_truths)
    ]
    raw_scores = [r["score"] for r in acc_results]
    return {
        "f1": sum(f1s) / n,
        "recall": sum(recalls) / n,
        "accuracy": sum(s / 4.0 for s in raw_scores) / n,
        "accuracy_raw": sum(raw_scores) / n,
    }
