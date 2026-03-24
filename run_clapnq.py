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
2.  SQLAlchemy-based INSERT / DELETE + Agent-based RETRIEVE with asyncio concurrency
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
    """Load train + dev answerable QA items (unanswerable items are excluded)."""
    base = config.CLAPNQ_DATA_DIR

    raw_answerable: List[Dict] = []
    for split in ("train", "dev"):
        raw_answerable.extend(
            _load_jsonl(os.path.join(base, split, f"clapnq_{split}_answerable.jsonl"))
        )

    qa_list: List[Dict] = []

    for item in raw_answerable:
        gold_answers: List[str] = []
        selected_sentences: List[str] = []
        for out in item.get("output", []):
            ans = out.get("answer", "").strip()
            if ans:
                gold_answers.append(ans)
            selected_sentences.extend(out.get("selected_sentences", []))
        if not gold_answers:
            continue

        qa_list.append(
            {
                "qa_id": str(item["id"]),
                "question": item["input"],
                "passages": item.get("passages", []),
                "gold_answers": gold_answers,
                "evidence_texts": selected_sentences
                if selected_sentences
                else gold_answers,
                "answerable": True,
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


def _bulk_delete(db_path: str) -> float:
    engine = create_engine(f"sqlite:///{db_path}")
    t0 = time.time()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM passages"))
        conn.execute(text("DELETE FROM passage_chunks"))
        conn.commit()
    elapsed = time.time() - t0
    engine.dispose()
    logger.info("Bulk delete: all records in %.2fs", elapsed)
    return elapsed


def _insert_qa_group(engine, qa: Dict) -> None:
    """Insert passages + chunks for one QA item."""
    qa_id = qa["qa_id"]
    with engine.connect() as conn:
        for pg in qa.get("passages", []):
            title = pg.get("title", "")
            content = pg.get("text", "")
            conn.execute(
                text(
                    "INSERT INTO passages (qa_id, title, content) VALUES (:a, :b, :c)"
                ),
                {"a": qa_id, "b": title, "c": content},
            )
            for ci, chunk in enumerate(_chunk_text(content)):
                conn.execute(
                    text(
                        "INSERT INTO passage_chunks "
                        "(qa_id, title, chunk_index, content) VALUES (:a, :b, :c, :d)"
                    ),
                    {"a": qa_id, "b": title, "c": ci, "d": chunk},
                )
        conn.commit()


def _delete_qa_group(engine, qa_id: str) -> None:
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM passages WHERE qa_id = :q"), {"q": qa_id})
        conn.execute(text("DELETE FROM passage_chunks WHERE qa_id = :q"), {"q": qa_id})
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
        "  CLAPNQ Evaluation  (sample_size=%d, concurrency=%d, mode=%s)"
        % (sample_size, config.CONCURRENCY, mode)
    )
    print("=" * 60)

    all_qa = _load_all_data()

    db_path = config.CLAPNQ_DB
    _init_db(db_path)

    if mode == "per-question":
        engine = create_engine(f"sqlite:///{db_path}")
        tracker = TokenTracker(config.TIKTOKEN_ENCODING)
        op = OperationTracker(tracker)
        agent = build_agent(db_path, tracker, verbose=verbose)
        accuracy_llm = get_llm()
        sem = asyncio.Semaphore(config.CONCURRENCY)

        rng = random.Random(42)
        retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))

        groups: Dict[str, List[Dict]] = defaultdict(list)
        for qa in retrieve_qa:
            groups[qa["qa_id"]].append(qa)

        answers: List[tuple] = []
        qa_order: List[Dict] = []

        async def _do_retrieve(qa: Dict):
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

        for qa_id, qa_list in groups.items():
            for qa in qa_list:
                _insert_qa_group(engine, qa)
            group_answers = await asyncio.gather(*[_do_retrieve(qa) for qa in qa_list])
            _delete_qa_group(engine, qa_id)
            for qa, ans in zip(qa_list, group_answers):
                answer, retrieved = (
                    (ans[0] or "", ans[1] or [])
                    if isinstance(ans, tuple) and len(ans) >= 2
                    else (ans or "", [])
                )
                qa_order.append(qa)
                answers.append((answer, retrieved))

        per_item: list = []
        all_retrieved: list[list[str]] = []
        evidences: list[list[str]] = []
        for qa, (answer, retrieved) in zip(qa_order, answers):
            golds = qa["gold_answers"]
            ev = qa.get("evidence_texts", golds)
            all_retrieved.append(retrieved)
            evidences.append(ev)
            per_item.append(
                {
                    "qa_id": qa["qa_id"],
                    "question": qa["question"],
                    "answerable": qa["answerable"],
                    "ground_truth": golds,
                    "evidence_texts": ev,
                    "prediction": answer,
                    "retrieved_texts": retrieved,
                    "f1": token_f1(answer, golds),
                    "recall": token_recall(answer, ev, retrieved_texts=retrieved),
                    "accuracy": accuracy(
                        accuracy_llm, qa["question"], answer, golds, "clapnq"
                    ),
                }
            )

        questions = [item["question"] for item in per_item]
        predictions = [item["prediction"] for item in per_item]
        ground_truths = [item["ground_truth"] for item in per_item]
        qa_metrics = compute_metrics(
            accuracy_llm,
            questions,
            predictions,
            ground_truths,
            "clapnq",
            evidences=evidences,
            retrieved_texts_list=all_retrieved,
        )

        ans_ques: dict = defaultdict(list)
        ans_preds: dict = defaultdict(list)
        ans_gts: dict = defaultdict(list)
        ans_evidences: dict = defaultdict(list)
        ans_retrieved: dict = defaultdict(list)
        for item in per_item:
            label = "answerable" if item["answerable"] else "unanswerable"
            ans_ques[label].append(item["question"])
            ans_preds[label].append(item["prediction"])
            ans_gts[label].append(item["ground_truth"])
            ans_evidences[label].append(item["evidence_texts"])
            ans_retrieved[label].append(item.get("retrieved_texts", []))
        split_metrics = {
            k: compute_metrics(
                accuracy_llm,
                ans_ques[k],
                ans_preds[k],
                ans_gts[k],
                "clapnq",
                evidences=ans_evidences[k],
                retrieved_texts_list=ans_retrieved[k],
            )
            for k in sorted(ans_preds)
        }

        report = {
            "dataset": "CLAPNQ",
            "sample_size": sample_size,
            "mode": mode,
            "concurrency": config.CONCURRENCY,
            "num_qa_total": len(all_qa),
            "num_retrieve": len(retrieve_qa),
            "qa_metrics": qa_metrics,
            "qa_metrics_by_answerability": split_metrics,
            "retrieve_metrics": op.summary("retrieve"),
            "per_item": per_item,
        }

        engine.dispose()
        _print_report(report)
        return report

    bulk_time = _bulk_insert(db_path, all_qa)

    tracker = TokenTracker(config.TIKTOKEN_ENCODING)
    op = OperationTracker(tracker)
    agent = build_agent(db_path, tracker, verbose=verbose)

    rng = random.Random(42)
    retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))

    sem = asyncio.Semaphore(config.CONCURRENCY)

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

    results = await _run_concurrent(
        [_do_retrieve(qa) for qa in retrieve_qa], "retrieve"
    )

    accuracy_llm = get_llm()
    questions: list[str] = []
    predictions: list[str] = []
    ground_truths: list[Union[str, List[str]]] = []
    evidences: list[list[str]] = []
    all_retrieved: list[list[str]] = []
    per_item: list[dict] = []
    for qa, res in zip(retrieve_qa, results):
        answer, retrieved = (
            (res[0] or "", res[1] or [])
            if isinstance(res, tuple) and len(res) >= 2
            else (res or "", [])
        )
        predictions.append(answer)
        questions.append(qa["question"])
        golds = qa["gold_answers"]
        ev = qa.get("evidence_texts", golds)
        ground_truths.append(golds)
        evidences.append(ev)
        all_retrieved.append(retrieved)
        per_item.append(
            {
                "qa_id": qa["qa_id"],
                "question": qa["question"],
                "answerable": qa["answerable"],
                "ground_truth": golds,
                "evidence_texts": ev,
                "prediction": answer,
                "retrieved_texts": retrieved,
                "f1": token_f1(answer, golds),
                "recall": token_recall(answer, ev, retrieved_texts=retrieved),
                "accuracy": accuracy(
                    accuracy_llm, qa["question"], answer, golds, "clapnq"
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
        "clapnq",
        evidences=evidences,
        retrieved_texts_list=all_retrieved,
    )

    ans_ques: dict = defaultdict(list)
    ans_preds: dict = defaultdict(list)
    ans_gts: dict = defaultdict(list)
    ans_evidences: dict = defaultdict(list)
    ans_retrieved: dict = defaultdict(list)
    for item in per_item:
        label = "answerable" if item["answerable"] else "unanswerable"
        ans_ques[label].append(item["question"])
        ans_preds[label].append(item["prediction"])
        ans_gts[label].append(item["ground_truth"])
        ans_evidences[label].append(item["evidence_texts"])
        ans_retrieved[label].append(item.get("retrieved_texts", []))
    split_metrics = {
        k: compute_metrics(
            accuracy_llm,
            ans_ques[k],
            ans_preds[k],
            ans_gts[k],
            "clapnq",
            evidences=ans_evidences[k],
            retrieved_texts_list=ans_retrieved[k],
        )
        for k in sorted(ans_preds)
    }

    report = {
        "dataset": "CLAPNQ",
        "sample_size": sample_size,
        "mode": mode,
        "concurrency": config.CONCURRENCY,
        "num_qa_total": len(all_qa),
        "num_retrieve": len(retrieve_qa),
        "bulk_insert_time": bulk_time,
        "bulk_delete_time": bulk_delete_time,
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
    print(f"  CLAPNQ Results  (mode={r.get('mode', 'bulk')})")
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
    if "bulk_insert_time" in r:
        print(f"  Bulk insert time: {r['bulk_insert_time']:.2f}s")
    if "bulk_delete_time" in r:
        print(f"  Bulk delete time: {r['bulk_delete_time']:.2f}s")

    for op_name in ("insert", "retrieve", "delete"):
        m = r.get(f"{op_name}_metrics", {})
        if m:
            print(
                f"  {op_name:>8}: n={m['count']}  "
                f"avg_time={m['avg_time']:.2f}s  total_time={m['total_time']:.2f}s  "
                f"avg_tokens={m['avg_tokens']:.0f}  total_tokens={m['total_tokens']}"
            )
    print("-" * 60)
