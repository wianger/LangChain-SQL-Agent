"""SyllabusQA evaluation pipeline (async-concurrent version).

1.  Load test.json + syllabus text files -> populate SQLite
2.  SQLAlchemy-based INSERT / DELETE + Agent-based RETRIEVE with asyncio concurrency
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


NO_ANSWER_PATTERNS = [
    "no answer",
    "unanswerable",
    "no/insufficient information",
    "not mentioned",
    "none",
    "n/a",
]


def _is_no_answer(answer: str) -> bool:
    lowered = answer.strip().lower()
    return any(p == lowered for p in NO_ANSWER_PATTERNS)


def _extract_evidence(item: Dict) -> List[str]:
    """Collect non-null answer_span_* and reasoning_step_* as evidence."""
    ev: List[str] = []
    for i in range(1, 6):
        val = item.get(f"answer_span_{i}")
        if val:
            ev.append(str(val).strip())
    for i in range(1, 6):
        val = item.get(f"reasoning_step_{i}")
        if val:
            ev.append(str(val).strip())
    return ev


def _load_test_qa() -> List[Dict]:
    results: List[Dict] = []
    for fname in ("test.json", "train.json", "val.json"):
        fpath = os.path.join(config.SYLLABUSQA_TEST_PATH, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw: List[Dict] = []
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = list(data.values())

        for item in raw:
            answer = str(item.get("answer", ""))
            qtype = str(item.get("question_type", "")).lower()
            if qtype == "no answer" or _is_no_answer(answer):
                continue
            item["evidence_texts"] = _extract_evidence(item) or [answer]
            results.append(item)
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


def _bulk_delete(db_path: str) -> float:
    engine = create_engine(f"sqlite:///{db_path}")
    t0 = time.time()

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM syllabus_chunks"))
        conn.execute(text("DELETE FROM syllabi"))
        conn.commit()

    elapsed = time.time() - t0
    engine.dispose()
    logger.info("Bulk delete: all records in %.2fs", elapsed)
    return elapsed


def _insert_syllabus_group(engine, name: str, meta: Dict) -> None:
    content = _load_syllabus_text(name)
    m = meta.get(name, {})
    with engine.connect() as conn:
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
                    "(syllabus_name, chunk_index, content) VALUES (:a,:b,:c)"
                ),
                {"a": name, "b": ci, "c": chunk},
            )
        conn.commit()


def _delete_syllabus_group(engine, name: str) -> None:
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM syllabi WHERE syllabus_name = :n"), {"n": name})
        conn.execute(
            text("DELETE FROM syllabus_chunks WHERE syllabus_name = :n"), {"n": name}
        )
        conn.commit()


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
    mode: str = "bulk",
) -> Dict[str, Any]:

    print("\n" + "=" * 60)
    print(
        "  SyllabusQA Evaluation  (sample_size=%d, concurrency=%d, mode=%s)"
        % (sample_size, config.CONCURRENCY, mode)
    )
    print("=" * 60)

    all_qa = _load_test_qa()

    db_path = config.SYLLABUSQA_DB
    _init_db(db_path)

    if mode == "per-question":
        tracker = TokenTracker(config.TIKTOKEN_ENCODING)
        op = OperationTracker(tracker)
        agent = build_agent(db_path, tracker, verbose=verbose)
        accuracy_llm = get_llm()
        sem = asyncio.Semaphore(config.CONCURRENCY)

        rng = random.Random(42)
        retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for qa in retrieve_qa:
            groups[qa["syllabus_name"]].append(qa)

        meta = _load_meta()
        engine = create_engine(f"sqlite:///{db_path}")

        total_setup_time = 0.0
        total_teardown_time = 0.0
        answer_map: Dict[tuple, tuple[str, list[str]]] = {}

        for syllabus_name, group_qas in groups.items():
            t0 = time.time()
            _insert_syllabus_group(engine, syllabus_name, meta)
            total_setup_time += time.time() - t0

            async def _do_retrieve_pq(qa):
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

            results = await _run_concurrent(
                [_do_retrieve_pq(qa) for qa in group_qas], "retrieve"
            )
            for qa, (answer, retrieved) in zip(group_qas, results):
                answer_map[(qa["syllabus_name"], qa["question"])] = (
                    answer or "",
                    retrieved or [],
                )

            t0 = time.time()
            _delete_syllabus_group(engine, syllabus_name)
            total_teardown_time += time.time() - t0

        engine.dispose()

        all_retrieved: list[list[str]] = []
        per_item: List[Dict] = []
        for qa in retrieve_qa:
            answer, retrieved = answer_map.get(
                (qa["syllabus_name"], qa["question"]), ("", [])
            )
            all_retrieved.append(retrieved)
            ev = qa.get("evidence_texts", [])
            ev = ev if ev else [qa["answer"]]
            ev_or_gt = ev if ev else [qa["answer"]]
            per_item.append(
                {
                    "question": qa["question"],
                    "syllabus": qa["syllabus_name"],
                    "question_type": qa.get("question_type", ""),
                    "ground_truth": qa["answer"],
                    "evidence_texts": ev,
                    "prediction": answer,
                    "retrieved_texts": retrieved,
                    "f1": token_f1(answer, qa["answer"]),
                    "recall": token_recall(answer, ev_or_gt, retrieved_texts=retrieved),
                    "accuracy": accuracy(
                        accuracy_llm, qa["question"], answer, qa["answer"], "SyllabusQA"
                    ),
                }
            )

        questions = [x["question"] for x in per_item]
        predictions = [x["prediction"] for x in per_item]
        ground_truths = [x["ground_truth"] for x in per_item]
        evidences = [x["evidence_texts"] for x in per_item]

        qa_metrics = compute_metrics(
            accuracy_llm,
            questions,
            predictions,
            ground_truths,
            "SyllabusQA",
            evidences=evidences,
            retrieved_texts_list=all_retrieved,
        )

        type_questions: dict = defaultdict(list)
        type_preds: dict = defaultdict(list)
        type_gts: dict = defaultdict(list)
        type_evidences: dict = defaultdict(list)
        type_retrieved: dict = defaultdict(list)
        for item in per_item:
            qtype = item.get("question_type", "unknown")
            type_questions[qtype].append(item["question"])
            type_preds[qtype].append(item["prediction"])
            type_gts[qtype].append(item["ground_truth"])
            ev = item.get("evidence_texts", [])
            type_evidences[qtype].append(ev if ev else [item["ground_truth"]])
            type_retrieved[qtype].append(item["retrieved_texts"])
        type_metrics = {
            t: compute_metrics(
                accuracy_llm,
                type_questions[t],
                type_preds[t],
                type_gts[t],
                "SyllabusQA",
                evidences=type_evidences[t],
                retrieved_texts_list=type_retrieved[t],
            )
            for t in sorted(type_preds)
        }

        report = {
            "dataset": "SyllabusQA",
            "sample_size": sample_size,
            "concurrency": config.CONCURRENCY,
            "mode": "per-question",
            "num_retrieve": len(retrieve_qa),
            "setup_time": total_setup_time,
            "teardown_time": total_teardown_time,
            "num_doc_groups": len(groups),
            "qa_metrics": qa_metrics,
            "qa_metrics_by_type": type_metrics,
            "retrieve_metrics": op.summary("retrieve"),
            "per_item": per_item,
        }
        _print_report(report)
        return report

    bulk_time = _bulk_insert(db_path, all_qa)

    tracker = TokenTracker(config.TIKTOKEN_ENCODING)
    op = OperationTracker(tracker)
    agent = build_agent(db_path, tracker, verbose=verbose)

    rng = random.Random(42)
    retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))
    meta = _load_meta()
    sem = asyncio.Semaphore(config.CONCURRENCY)

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

    results = await _run_concurrent(
        [_do_retrieve(qa) for qa in retrieve_qa], "retrieve"
    )

    accuracy_llm = get_llm()
    questions: list[str] = []
    predictions: list[str] = []
    ground_truths: list[str] = []
    evidences: list[list[str]] = []
    all_retrieved: list[list[str]] = []
    per_item: list[dict] = []
    for qa, (answer, retrieved) in zip(retrieve_qa, results):
        questions.append(qa["question"])
        answer = answer or ""
        predictions.append(answer)
        gt = qa["answer"]
        ground_truths.append(gt)
        ev = qa.get("evidence_texts", [])
        ev_or_gt = ev if ev else [gt]
        evidences.append(ev_or_gt)
        all_retrieved.append(retrieved or [])
        per_item.append(
            {
                "question": qa["question"],
                "syllabus": qa["syllabus_name"],
                "question_type": qa.get("question_type", ""),
                "ground_truth": gt,
                "evidence_texts": ev,
                "prediction": answer,
                "retrieved_texts": retrieved or [],
                "f1": token_f1(answer, gt),
                "recall": token_recall(
                    answer, ev_or_gt, retrieved_texts=retrieved or []
                ),
                "accuracy": accuracy(
                    accuracy_llm, qa["question"], answer, gt, "SyllabusQA"
                ),
            }
        )

    # ── DELETE ──────────────────────────────────────────────────────────────
    bulk_delete_time = _bulk_delete(db_path)

    # ── Metrics ─────────────────────────────────────────────────────────────
    qa_metrics = compute_metrics(
        accuracy_llm,
        questions,
        predictions,
        ground_truths,
        "SyllabusQA",
        evidences=evidences,
        retrieved_texts_list=all_retrieved,
    )

    type_questions: dict = defaultdict(list)
    type_preds: dict = defaultdict(list)
    type_gts: dict = defaultdict(list)
    type_evidences: dict = defaultdict(list)
    type_retrieved: dict = defaultdict(list)
    for item in per_item:
        qtype = item.get("question_type", "unknown")
        type_questions[qtype].append(item["question"])
        type_preds[qtype].append(item["prediction"])
        type_gts[qtype].append(item["ground_truth"])
        ev = item.get("evidence_texts", [])
        type_evidences[qtype].append(ev if ev else [item["ground_truth"]])
        type_retrieved[qtype].append(item["retrieved_texts"])
    type_metrics = {
        t: compute_metrics(
            accuracy_llm,
            type_questions[t],
            type_preds[t],
            type_gts[t],
            "SyllabusQA",
            evidences=type_evidences[t],
            retrieved_texts_list=type_retrieved[t],
        )
        for t in sorted(type_preds)
    }

    report = {
        "dataset": "SyllabusQA",
        "sample_size": sample_size,
        "concurrency": config.CONCURRENCY,
        "num_retrieve": len(retrieve_qa),
        "bulk_insert_time": bulk_time,
        "bulk_delete_time": bulk_delete_time,
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
    mode = r.get("mode", "bulk")
    print("\n" + "-" * 60)
    print(f"  SyllabusQA Results  (mode={mode})")
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
    if "bulk_insert_time" in r:
        print(f"  Bulk insert time: {r['bulk_insert_time']:.2f}s")
    if "bulk_delete_time" in r:
        print(f"  Bulk delete time: {r['bulk_delete_time']:.2f}s")
    if "setup_time" in r:
        print(f"  Doc groups: {r.get('num_doc_groups', 0)}")
        print(f"  Setup time: {r['setup_time']:.2f}s")
        print(f"  Teardown time: {r['teardown_time']:.2f}s")

    for op_name in ("insert", "retrieve", "delete"):
        m = r.get(f"{op_name}_metrics", {})
        if m:
            print(
                f"  {op_name:>8}: n={m['count']}  "
                f"avg_time={m['avg_time']:.2f}s  total_time={m['total_time']:.2f}s  "
                f"avg_tokens={m['avg_tokens']:.0f}  total_tokens={m['total_tokens']}"
            )
    print("-" * 60)
