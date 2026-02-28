"""SyllabusQA evaluation pipeline (async-concurrent version).

1.  Load test.json + syllabus text files -> populate SQLite
2.  Agent-based INSERT / RETRIEVE / DELETE with asyncio concurrency
3.  Report F1, recall, accuracy + time / token costs
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List

from sqlalchemy import create_engine, text
from tqdm import tqdm

import config
from metrics import accuracy, compute_metrics, token_f1, token_recall
from sql_agent import arun_agent, build_agent, get_llm
from token_tracker import OperationTracker, TokenTracker

logger = logging.getLogger(__name__)

# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS syllabi (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    syllabus_name   TEXT,
    course          TEXT,
    major           TEXT,
    area            TEXT,
    university      TEXT,
    num_pages       INTEGER,
    content         TEXT
);

CREATE TABLE IF NOT EXISTS syllabus_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    syllabus_name   TEXT,
    chunk_index     INTEGER,
    content         TEXT
);
"""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ── Data loading helpers ─────────────────────────────────────────────────────


def _load_meta() -> Dict[str, Dict]:
    meta: Dict[str, Dict] = {}
    with open(config.SYLLABI_META_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row["name"]] = {
                "course": row.get("course", ""),
                "major": row.get("major", ""),
                "area": row.get("area", ""),
                "university": row.get("university", ""),
                "num_pages": int(row.get("num_pages", 0) or 0),
            }
    return meta


def _load_syllabus_text(name: str) -> str:
    path = os.path.join(config.SYLLABI_TEXT_DIR, name + ".txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _chunk_text(text_content: str) -> List[str]:
    if not text_content:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text_content):
        end = start + CHUNK_SIZE
        chunks.append(text_content[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


def _load_test_qa() -> List[Dict]:
    results: List[Dict] = []
    for fname in ("test.json", "train.json", "val.json"):
        fpath = os.path.join(config.SYLLABUSQA_TEST_PATH, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            results.extend(data)
        elif isinstance(data, dict):
            results.extend(data.values())
    return results


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
    meta = _load_meta()
    needed = {q["syllabus_name"] for q in qa_data}

    engine = create_engine(f"sqlite:///{db_path}")
    t0 = time.time()

    with engine.connect() as conn:
        for name in sorted(needed):
            content = _load_syllabus_text(name)
            m = meta.get(name, {})
            conn.execute(
                text(
                    "INSERT INTO syllabi "
                    "(syllabus_name, course, major, area, university, num_pages, content) "
                    "VALUES (:a,:b,:c,:d,:e,:f,:g)"
                ),
                {
                    "a": name,
                    "b": m.get("course", ""),
                    "c": m.get("major", ""),
                    "d": m.get("area", ""),
                    "e": m.get("university", ""),
                    "f": m.get("num_pages", 0),
                    "g": content,
                },
            )
            for ci, chunk in enumerate(_chunk_text(content)):
                conn.execute(
                    text(
                        "INSERT INTO syllabus_chunks "
                        "(syllabus_name, chunk_index, content) "
                        "VALUES (:a,:b,:c)"
                    ),
                    {"a": name, "b": ci, "c": chunk},
                )
        conn.commit()

    elapsed = time.time() - t0
    engine.dispose()
    logger.info("Bulk insert: %d syllabi in %.2fs", len(needed), elapsed)
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
        "  SyllabusQA Evaluation  (sample_size=%d, concurrency=%d)"
        % (sample_size, config.CONCURRENCY)
    )
    print("=" * 60)

    all_qa = _load_test_qa()

    db_path = config.SYLLABUSQA_DB
    _init_db(db_path)
    bulk_time = _bulk_insert(db_path, all_qa)

    tracker = TokenTracker(config.TIKTOKEN_ENCODING)
    op = OperationTracker(tracker)
    agent = build_agent(db_path, tracker, verbose=verbose)

    rng = random.Random(42)
    retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))

    meta = _load_meta()
    syllabi_names = list({q["syllabus_name"] for q in all_qa})
    insert_names = rng.sample(syllabi_names, min(sample_size, len(syllabi_names)))

    sem = asyncio.Semaphore(config.CONCURRENCY)

    # ── INSERT ──────────────────────────────────────────────────────────────
    print(
        "\n[INSERT] %d chunks (concurrency=%d)..."
        % (len(insert_names), config.CONCURRENCY)
    )

    async def _do_insert(name):
        content_snippet = _load_syllabus_text(name)[:300].replace("'", "''")
        prompt = (
            f"Insert a new record into the syllabus_chunks table with: "
            f"syllabus_name='{name}', chunk_index=999, "
            f"content='{content_snippet}'"
        )
        async with sem:
            with op.track("insert"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_insert(n) for n in insert_names], "insert")

    # ── RETRIEVE ────────────────────────────────────────────────────────────
    print(
        "\n[RETRIEVE] %d questions (concurrency=%d)..."
        % (len(retrieve_qa), config.CONCURRENCY)
    )

    async def _do_retrieve(qa):
        prompt = (
            f"Using the syllabus data for '{qa['syllabus_name']}' stored in the database, "
            f"answer the following question. "
            f"Search in both 'syllabi' and 'syllabus_chunks' tables. "
            f"If the answer cannot be found, reply 'No/insufficient information'.\n\n"
            f"Question: {qa['question']}"
        )
        async with sem:
            with op.track("retrieve"):
                return await arun_agent(agent, prompt)

    answers = await _run_concurrent(
        [_do_retrieve(qa) for qa in retrieve_qa], "retrieve"
    )

    accuracy_llm = get_llm()
    questions: list[str] =[]
    predictions: list[str] = []
    ground_truths: list[str] = []
    per_item: list[dict] = []
    for qa, answer in zip(retrieve_qa, answers):
        questions.append(qa["question"])
        answer = answer or ""
        predictions.append(answer)
        gt = qa["answer"]
        ground_truths.append(gt)
        per_item.append(
            {
                "question": qa["question"],
                "syllabus": qa["syllabus_name"],
                "question_type": qa.get("question_type", ""),
                "ground_truth": gt,
                "prediction": answer,
                "f1": token_f1(answer, gt),
                "recall": token_recall(answer, gt),
                "accuracy": accuracy(accuracy_llm,qa["question"],answer, gt, "SyllabusQA"),
            }
        )

    # ── DELETE ──────────────────────────────────────────────────────────────
    print(
        "\n[DELETE] %d records (concurrency=%d)..."
        % (len(insert_names), config.CONCURRENCY)
    )

    async def _do_delete(name):
        prompt = (
            f"Delete the record from syllabus_chunks where "
            f"syllabus_name = '{name}' AND chunk_index = 999."
        )
        async with sem:
            with op.track("delete"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_delete(n) for n in insert_names], "delete")

    # ── Metrics ─────────────────────────────────────────────────────────────
    qa_metrics = compute_metrics(accuracy_llm, questions, predictions, ground_truths, "SyllabusQA")

    type_questions: dict = defaultdict(list)
    type_preds: dict = defaultdict(list)
    type_gts: dict = defaultdict(list)
    for item in per_item:
        qtype = item.get("question_type", "unknown")
        type_questions[qtype].append(item["question"])
        type_preds[qtype].append(item["prediction"])
        type_gts[qtype].append(item["ground_truth"])
    type_metrics = {
        t: compute_metrics(accuracy_llm, type_questions[t], type_preds[t], type_gts[t], "SyllabusQA")
        for t in sorted(type_preds)
    }

    report = {
        "dataset": "SyllabusQA",
        "sample_size": sample_size,
        "concurrency": config.CONCURRENCY,
        "num_retrieve": len(retrieve_qa),
        "num_insert": len(insert_names),
        "num_delete": len(insert_names),
        "bulk_insert_time": bulk_time,
        "qa_metrics": qa_metrics,
        "qa_metrics_by_type": type_metrics,
        "insert_metrics": op.summary("insert"),
        "retrieve_metrics": op.summary("retrieve"),
        "delete_metrics": op.summary("delete"),
        "per_item": per_item,
    }

    _print_report(report)
    return report


def _print_report(r: Dict) -> None:
    print("\n" + "-" * 60)
    print("  SyllabusQA Results")
    print("-" * 60)
    qm = r["qa_metrics"]
    print(f"  F1:       {qm['f1']:.4f}")
    print(f"  Recall:   {qm['recall']:.4f}")
    print(f"  Accuracy: {qm['accuracy']:.4f}")
    print()
    for t, m in r.get("qa_metrics_by_type", {}).items():
        print(
            f"  [{t}]  F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Acc={m['accuracy']:.4f}"
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
