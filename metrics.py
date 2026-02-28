"""Evaluation metrics for QA: token-level F1, recall, and accuracy (substring)."""

from __future__ import annotations

import collections
import json
import re
import string
from collections import Counter
from typing import Dict, List, Union
from langchain_core.messages import HumanMessage, SystemMessage

REFUSAL_KEYWORDS = [
    "not mentioned",
    "no information",
    "cannot be answered",
    "none",
    "unknown",
    "don't know",
    "No/insufficient information"
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
def _token_recall_single(prediction: str, ground_truth: str) -> float:
    if _is_refusal(prediction) and _is_refusal(ground_truth):
        return 1.0
    pred_tokens = _get_tokens(prediction)
    gt_tokens = _get_tokens(ground_truth)
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    return sum(common.values()) / len(gt_tokens)


def token_recall(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    if isinstance(ground_truth, list):
        return max(
            (_token_recall_single(prediction, gt) for gt in ground_truth), default=0.0
        )
    return _token_recall_single(prediction, ground_truth)


# ── Accuracy (substring containment) ────────────────────────────────────────
def llm_grader(
    llm,
    question: str,
    gold_answer: str,
    response: str,
    dataset_name: str = "Locomo",
) -> float:

    # 1. 根据 dataset_name 路由选择 Prompt
    if "Locomo" in dataset_name.lower():
        system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """
        ACCURACY_PROMPT = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
    """
    else:
        # 通用 Prompt 或其他数据集的 Prompt
        system_prompt = """
        You are an expert grader that determines if an AI-generated answer matches the gold standard (ground truth) answer for a given question.
        """
        ACCURACY_PROMPT = f"""
        Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
            (1) A question
            (2) A 'gold' (ground truth) answer
            (3) A generated answer

        Grading rules:
        - If the generated answer correctly encompasses the core semantic meaning or facts of the gold answer, grade it as CORRECT.
        - If the generated answer contradicts the gold answer or misses the key factual information, it is WRONG.

        Question: {question}
        Gold answer: {gold_answer}
        Generated answer: {response}

        First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
        Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
        """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=ACCURACY_PROMPT),
    ]
    resp = llm.invoke(messages)
    content = resp.content

    try:
        result = json.loads(content)
        label = result.get("is_correct", result.get("label", "WRONG"))
        return 1.0 if label.strip().lower() == "correct" else 0.0
    except json.JSONDecodeError:
        # 容错：防止 LLM 没按格式输出 JSON
        return 1.0 if "CORRECT" in content.upper() else 0.0


def accuracy(
    llm,
    question: str,
    prediction: str,
    ground_truth: Union[str, List[str]],
    datasetname,
) -> float:
    """1.0 if any gold answer (lowered) is a substring of prediction (lowered)."""
    if isinstance(ground_truth, list):
        return max(
            (
                llm_grader(llm, question, prediction, gt, datasetname)
                for gt in ground_truth
            ),
            default=0.0,
        )
    return llm_grader(llm, question, prediction, ground_truth, datasetname)


# ── Aggregate ────────────────────────────────────────────────────────────────
def compute_metrics(
    llm,
    questions: List[str],
    predictions: List[str],
    ground_truths: List[Union[str, List[str]]],
    dataset_name: str = "Locomo",
) -> Dict[str, float]:
    """Return averaged F1, recall, and accuracy over (pred, gt) pairs."""
    assert len(predictions) == len(ground_truths)
    n = len(predictions)
    if n == 0:
        return {"f1": 0.0, "recall": 0.0, "accuracy": 0.0}
    f1s = [token_f1(p, g) for p, g in zip(predictions, ground_truths)]
    recalls = [token_recall(p, g) for p, g in zip(predictions, ground_truths)]
    accs = [
        accuracy(llm, q, p, g, dataset_name)
        for q, p, g in zip(questions, predictions, ground_truths)
    ]
    return {
        "f1": sum(f1s) / n,
        "recall": sum(recalls) / n,
        "accuracy": sum(accs) / n,
    }
