"""CLAPNQ evaluation pipeline (async-concurrent version).

CLAPNQ (Cohesive Long-form Answers from Passages in Natural Questions)
is a RAG benchmark where each question is paired with a Wikipedia passage.

Data format (annotated_data/):
  - id, input (question), passages [{title, text, sentences}],
    output [{answer, selected_sentences, meta}]
  - answerable items have gold long-form answers
  - unanswerable items have empty answers

Pipeline
--------
1.  Load dev answerable + unanswerable JSONL -> populate SQLite
2.  Agent-based INSERT / RETRIEVE / DELETE with asyncio concurrency
3.  Report F1, recall, accuracy + time / token costs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Union

from sqlalchemy import create_engine, text
from tqdm import tqdm

import config
from metrics import accuracy, compute_metrics, token_f1, token_recall
from sql_agent import arun_agent, build_agent, get_llm
from token_tracker import OperationTracker, TokenTracker

logger = logging.getLogger(__name__)

# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS passages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    qa_id       TEXT,
    title       TEXT,
    content     TEXT
);

CREATE TABLE IF NOT EXISTS passage_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    qa_id       TEXT,
    title       TEXT,
    chunk_index INTEGER,
    content     TEXT
);
"""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ── Data loading helpers ─────────────────────────────────────────────────────


def _load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_all_data() -> List[Dict]:
    """Load train + dev (answerable + unanswerable) as a unified QA list."""
    base = config.CLAPNQ_DATA_DIR

    raw_answerable: List[Dict] = []
    raw_unanswerable: List[Dict] = []
    for split in ("train", "dev"):
        raw_answerable.extend(
            _load_jsonl(os.path.join(base, split, f"clapnq_{split}_answerable.jsonl"))
        )
        raw_unanswerable.extend(
            _load_jsonl(os.path.join(base, split, f"clapnq_{split}_unanswerable.jsonl"))
        )

    qa_list: List[Dict] = []

    for item in raw_answerable:
        gold_answers: List[str] = []
        for out in item.get("output", []):
            ans = out.get("answer", "").strip()
            if ans:
                gold_answers.append(ans)
        if not gold_answers:
            gold_answers = ["unanswerable"]

        qa_list.append(
            {
                "qa_id": str(item["id"]),
                "question": item["input"],
                "passages": item.get("passages", []),
                "gold_answers": gold_answers,
                "answerable": True,
            }
        )

    for item in raw_unanswerable:
        qa_list.append(
            {
                "qa_id": str(item["id"]),
                "question": item["input"],
                "passages": item.get("passages", []),
                "gold_answers": ["unanswerable"],
                "answerable": False,
            }
        )

    return qa_list


def _chunk_text(text_content: str) -> List[str]:
    if not text_content:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text_content):
        end = start + CHUNK_SIZE
        chunks.append(text_content[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


# ── Database operations ──────────────────────────────────────────────────────


def _init_db(db_path: str) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA busy_timeout=10000"))
        for stmt in SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
    engine.dispose()


def _bulk_insert(db_path: str, qa_data: List[Dict]) -> float:
    engine = create_engine(f"sqlite:///{db_path}")
    t0 = time.time()
    total_chunks = 0

    with engine.connect() as conn:
        for qa in tqdm(qa_data, desc="loading passages"):
            qa_id = qa["qa_id"]
            for pg in qa.get("passages", []):
                title = pg.get("title", "")
                content = pg.get("text", "")

                conn.execute(
                    text(
                        "INSERT INTO passages (qa_id, title, content) "
                        "VALUES (:a, :b, :c)"
                    ),
                    {"a": qa_id, "b": title, "c": content},
                )

                for ci, chunk in enumerate(_chunk_text(content)):
                    conn.execute(
                        text(
                            "INSERT INTO passage_chunks "
                            "(qa_id, title, chunk_index, content) "
                            "VALUES (:a, :b, :c, :d)"
                        ),
                        {"a": qa_id, "b": title, "c": ci, "d": chunk},
                    )
                    total_chunks += 1

        conn.commit()

    elapsed = time.time() - t0
    engine.dispose()
    logger.info(
        "Bulk insert: %d passages, %d chunks in %.2fs",
        len(qa_data),
        total_chunks,
        elapsed,
    )
    return elapsed


# ── Async helpers ────────────────────────────────────────────────────────────


async def _run_concurrent(coros: list, desc: str) -> list:
    pbar = tqdm(total=len(coros), desc=desc)
    results: list = [None] * len(coros)

    async def _wrap(idx: int, coro):
        results[idx] = await coro
        pbar.update(1)

    await asyncio.gather(*[_wrap(i, c) for i, c in enumerate(coros)])
    pbar.close()
    return results


# ── Experiment runner ────────────────────────────────────────────────────────


async def run_experiment(
    sample_size: int = config.SMALL_SAMPLE_SIZE,
    verbose: bool = False,
) -> Dict[str, Any]:

    print("\n" + "=" * 60)
    print(
        "  CLAPNQ Evaluation  (sample_size=%d, concurrency=%d)"
        % (sample_size, config.CONCURRENCY)
    )
    print("=" * 60)

    all_qa = _load_all_data()

    db_path = config.CLAPNQ_DB
    _init_db(db_path)
    bulk_time = _bulk_insert(db_path, all_qa)

    tracker = TokenTracker(config.TIKTOKEN_ENCODING)
    op = OperationTracker(tracker)
    agent = build_agent(db_path, tracker, verbose=verbose)

    rng = random.Random(42)
    retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))

    qa_ids = list({q["qa_id"] for q in all_qa})
    insert_ids = rng.sample(qa_ids, min(sample_size, len(qa_ids)))

    sem = asyncio.Semaphore(config.CONCURRENCY)

    # ── INSERT ──────────────────────────────────────────────────────────────
    print(
        "\n[INSERT] %d records (concurrency=%d)..."
        % (len(insert_ids), config.CONCURRENCY)
    )

    async def _do_insert(qa_id):
        prompt = (
            f"Insert a new record into the passage_chunks table with: "
            f"qa_id='{qa_id}', title='test', chunk_index=9999, "
            f"content='[test insert placeholder for {qa_id}]'"
        )
        async with sem:
            with op.track("insert"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_insert(qid) for qid in insert_ids], "insert")

    # ── RETRIEVE ────────────────────────────────────────────────────────────
    print(
        "\n[RETRIEVE] %d questions (concurrency=%d)..."
        % (len(retrieve_qa), config.CONCURRENCY)
    )

    async def _do_retrieve(qa):
        title_hint = ""
        if qa["passages"]:
            title_hint = f" The relevant passage is titled '{qa['passages'][0].get('title', '')}'."
        prompt = (
            f"Using the passage data stored in the database for "
            f"qa_id '{qa['qa_id']}',{title_hint} "
            f"answer the following question with a concise, cohesive answer. "
            f"Search in both 'passages' and 'passage_chunks' tables. "
            f"If the answer cannot be found, reply 'unanswerable'.\n\n"
            f"Question: {qa['question']}"
        )
        async with sem:
            with op.track("retrieve"):
                return await arun_agent(agent, prompt)

    answers = await _run_concurrent(
        [_do_retrieve(qa) for qa in retrieve_qa], "retrieve"
    )

    accuracy_llm = get_llm()
    questions: list[str] = []
    predictions: list[str] = []
    ground_truths: list[Union[str, List[str]]] = []
    per_item: list[dict] = []
    for qa, answer in zip(retrieve_qa, answers):
        answer = answer or ""
        predictions.append(answer)
        questions.append(qa["question"])
        golds = qa["gold_answers"]
        ground_truths.append(golds)
        per_item.append(
            {
                "qa_id": qa["qa_id"],
                "question": qa["question"],
                "answerable": qa["answerable"],
                "ground_truth": golds,
                "prediction": answer,
                "f1": token_f1(answer, golds),
                "recall": token_recall(answer, golds),
                "accuracy": accuracy(
                    accuracy_llm, qa["question"], answer, golds, "clapnq"
                ),
            }
        )

    # ── DELETE ──────────────────────────────────────────────────────────────
    print(
        "\n[DELETE] %d records (concurrency=%d)..."
        % (len(insert_ids), config.CONCURRENCY)
    )

    async def _do_delete(qa_id):
        prompt = (
            f"Delete the record from passage_chunks where "
            f"qa_id = '{qa_id}' AND chunk_index = 9999."
        )
        async with sem:
            with op.track("delete"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_delete(qid) for qid in insert_ids], "delete")

    # ── Metrics ─────────────────────────────────────────────────────────────
    qa_metrics = compute_metrics(
        accuracy_llm, questions, predictions, ground_truths, "clapnq"
    )

    ans_ques: dict = defaultdict(list)
    ans_preds: dict = defaultdict(list)
    ans_gts: dict = defaultdict(list)
    for item in per_item:
        label = "answerable" if item["answerable"] else "unanswerable"
        ans_ques[label].append(item["question"])
        ans_preds[label].append(item["prediction"])
        ans_gts[label].append(item["ground_truth"])
    split_metrics = {
        k: compute_metrics(
            accuracy_llm, ans_ques[k], ans_preds[k], ans_gts[k], "clapnq"
        )
        for k in sorted(ans_preds)
    }

    report = {
        "dataset": "CLAPNQ",
        "sample_size": sample_size,
        "concurrency": config.CONCURRENCY,
        "num_qa_total": len(all_qa),
        "num_retrieve": len(retrieve_qa),
        "num_insert": len(insert_ids),
        "num_delete": len(insert_ids),
        "bulk_insert_time": bulk_time,
        "qa_metrics": qa_metrics,
        "qa_metrics_by_answerability": split_metrics,
        "insert_metrics": op.summary("insert"),
        "retrieve_metrics": op.summary("retrieve"),
        "delete_metrics": op.summary("delete"),
        "per_item": per_item,
    }

    _print_report(report)
    return report


def _print_report(r: Dict) -> None:
    print("\n" + "-" * 60)
    print("  CLAPNQ Results")
    print("-" * 60)
    qm = r["qa_metrics"]
    print(f"  F1:       {qm['f1']:.4f}")
    print(f"  Recall:   {qm['recall']:.4f}")
    print(f"  Accuracy: {qm['accuracy']:.4f}")
    print(f"  Total QA: {r['num_qa_total']}")
    print()
    for k, m in r.get("qa_metrics_by_answerability", {}).items():
        print(
            f"  [{k}]  F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Acc={m['accuracy']:.4f}"
        )
    print()
    print(f"  Bulk insert time: {r['bulk_insert_time']:.2f}s")

    for op_name in ("insert", "retrieve", "delete"):
        m = r.get(f"{op_name}_metrics", {})
        if m:
            print(
                f"  {op_name:>8}: n={m['count']}  "
                f"avg_time={m['avg_time']:.2f}s  total_time={m['total_time']:.2f}s  "
                f"avg_tokens={m['avg_tokens']:.0f}  total_tokens={m['total_tokens']}"
            )
    print("-" * 60)
