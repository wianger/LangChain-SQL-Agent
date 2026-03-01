"""QASPER evaluation pipeline (async-concurrent version).

QASPER is a QA dataset over NLP research papers.  Each paper contains:
  - title, abstract
  - full_text  (section_name[] + paragraphs[][])
  - qas        (question[], answers[])

Answer types:  extractive  |  free_form  |  yes_no  |  unanswerable

Pipeline
--------
1.  Load test.json (JSONL) -> populate SQLite with paper sections
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
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import create_engine, text
from tqdm import tqdm

import config
from metrics import accuracy, compute_metrics, token_f1, token_recall
from sql_agent import arun_agent, build_agent,get_llm
from token_tracker import OperationTracker, TokenTracker

logger = logging.getLogger(__name__)

# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    TEXT UNIQUE,
    title       TEXT,
    abstract    TEXT
);

CREATE TABLE IF NOT EXISTS sections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id      TEXT,
    section_index INTEGER,
    section_name  TEXT,
    content       TEXT
);

CREATE TABLE IF NOT EXISTS section_chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id      TEXT,
    section_index INTEGER,
    chunk_index   INTEGER,
    content       TEXT
);
"""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ── Data loading helpers ─────────────────────────────────────────────────────

def _load_papers() -> List[Dict]:
    papers: List[Dict] = []
    test_file = os.path.join(config.QASPER_PATH, "test.json")
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    train_file = os.path.join(config.QASPER_PATH, "train.json")
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    valid_file = os.path.join(config.QASPER_PATH, "validation.json")
    with open(valid_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def _get_gold_answer(answer_obj: Dict) -> tuple[str, str]:
    """Extract a single canonical answer string and its type from one annotator answer.

    Returns (answer_text, answer_type).
    """
    if answer_obj.get("unanswerable"):
        return "unanswerable", "unanswerable"
    if answer_obj.get("yes_no") is not None:
        return "yes" if answer_obj["yes_no"] else "no", "yes_no"
    if answer_obj.get("extractive_spans"):
        return " ".join(answer_obj["extractive_spans"]).strip(), "extractive"
    if answer_obj.get("free_form_answer"):
        return answer_obj["free_form_answer"].strip(), "free_form"
    return "", "empty"


def _extract_qa(papers: List[Dict]) -> List[Dict]:
    """Flatten papers into a list of QA dicts.

    For each question we collect ALL annotator answers and pick the majority
    answer type.  The gold answers list is kept for multi-reference F1.
    """
    qa_list: List[Dict] = []
    for paper in papers:
        pid = paper["id"]
        title = paper.get("title", "")
        qas = paper["qas"]
        questions = qas["question"]
        question_ids = qas["question_id"]
        answers_per_q = qas["answers"]

        for idx in range(len(questions)):
            annotator_answers = answers_per_q[idx]["answer"]
            gold_texts: List[str] = []
            type_counts: Dict[str, int] = defaultdict(int)

            for ann in annotator_answers:
                txt, atype = _get_gold_answer(ann)
                if txt:
                    gold_texts.append(txt)
                type_counts[atype] += 1

            majority_type = max(type_counts, key=lambda k: type_counts[k]) if type_counts else "unknown"

            if not gold_texts:
                gold_texts = ["unanswerable"]
                majority_type = "unanswerable"

            qa_list.append({
                "paper_id": pid,
                "title": title,
                "question_id": question_ids[idx],
                "question": questions[idx],
                "gold_answers": gold_texts,
                "answer_type": majority_type,
            })
    return qa_list


def _build_sections(paper: Dict) -> List[Dict[str, Any]]:
    """Return list of {section_index, section_name, content} for one paper."""
    ft = paper.get("full_text", {})
    section_names = ft.get("section_name", [])
    paragraphs = ft.get("paragraphs", [])
    sections: List[Dict[str, Any]] = []
    for i, (sname, paras) in enumerate(zip(section_names, paragraphs)):
        content = "\n".join(paras) if isinstance(paras, list) else str(paras)
        sections.append({
            "section_index": i,
            "section_name": sname,
            "content": content,
        })
    return sections


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


def _bulk_insert(db_path: str, papers: List[Dict]) -> float:
    engine = create_engine(f"sqlite:///{db_path}")
    t0 = time.time()
    total_chunks = 0

    with engine.connect() as conn:
        for paper in tqdm(papers, desc="loading papers"):
            pid = paper["id"]
            conn.execute(
                text(
                    "INSERT OR IGNORE INTO papers (paper_id, title, abstract) "
                    "VALUES (:a, :b, :c)"
                ),
                {"a": pid, "b": paper.get("title", ""), "c": paper.get("abstract", "")},
            )

            for sec in _build_sections(paper):
                conn.execute(
                    text(
                        "INSERT INTO sections (paper_id, section_index, section_name, content) "
                        "VALUES (:a, :b, :c, :d)"
                    ),
                    {
                        "a": pid,
                        "b": sec["section_index"],
                        "c": sec["section_name"],
                        "d": sec["content"],
                    },
                )
                for ci, chunk in enumerate(_chunk_text(sec["content"])):
                    conn.execute(
                        text(
                            "INSERT INTO section_chunks "
                            "(paper_id, section_index, chunk_index, content) "
                            "VALUES (:a, :b, :c, :d)"
                        ),
                        {"a": pid, "b": sec["section_index"], "c": ci, "d": chunk},
                    )
                    total_chunks += 1

        conn.commit()

    elapsed = time.time() - t0
    engine.dispose()
    logger.info(
        "Bulk insert: %d papers, %d chunks in %.2fs",
        len(papers), total_chunks, elapsed,
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
    print("  QASPER Evaluation  (sample_size=%d, concurrency=%d)" % (sample_size, config.CONCURRENCY))
    print("=" * 60)

    papers = _load_papers()
    all_qa = _extract_qa(papers)

    db_path = config.QASPER_DB
    _init_db(db_path)
    bulk_time = _bulk_insert(db_path, papers)

    tracker = TokenTracker(config.TIKTOKEN_ENCODING)
    op = OperationTracker(tracker)
    agent = build_agent(db_path, tracker, verbose=verbose)

    rng = random.Random(42)
    retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))

    paper_ids = list({q["paper_id"] for q in all_qa})
    insert_papers = rng.sample(paper_ids, min(sample_size, len(paper_ids)))

    sem = asyncio.Semaphore(config.CONCURRENCY)

    # ── INSERT ──────────────────────────────────────────────────────────────
    print("\n[INSERT] %d records (concurrency=%d)..." % (len(insert_papers), config.CONCURRENCY))

    async def _do_insert(pid):
        prompt = (
            f"Insert a new record into the section_chunks table with: "
            f"paper_id='{pid}', section_index=9999, chunk_index=0, "
            f"content='[test insert placeholder for {pid}]'"
        )
        async with sem:
            with op.track("insert"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_insert(p) for p in insert_papers], "insert")

    # ── RETRIEVE ────────────────────────────────────────────────────────────
    print("\n[RETRIEVE] %d questions (concurrency=%d)..." % (len(retrieve_qa), config.CONCURRENCY))

    async def _do_retrieve(qa):
        prompt = (
            f"Using the research paper data stored in the database for "
            f"paper_id '{qa['paper_id']}' (title: \"{qa['title']}\"), "
            f"answer the following question. "
            f"Search in the 'papers', 'sections', and 'section_chunks' tables. "
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
        per_item.append({
            "paper_id": qa["paper_id"],
            "question_id": qa["question_id"],
            "question": qa["question"],
            "answer_type": qa["answer_type"],
            "ground_truth": golds,
            "prediction": answer,
            "f1": token_f1(answer, golds),
            "recall": token_recall(answer, golds),
            "accuracy": accuracy(accuracy_llm, qa["question"], answer, golds,"QASPER"),
        })

    # ── DELETE ──────────────────────────────────────────────────────────────
    print("\n[DELETE] %d records (concurrency=%d)..." % (len(insert_papers), config.CONCURRENCY))

    async def _do_delete(pid):
        prompt = (
            f"Delete the record from section_chunks where "
            f"paper_id = '{pid}' AND section_index = 9999."
        )
        async with sem:
            with op.track("delete"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_delete(p) for p in insert_papers], "delete")

    # ── Metrics ─────────────────────────────────────────────────────────────
    qa_metrics = compute_metrics(accuracy_llm, questions, predictions, ground_truths, "QASPER")

    type_questions: dict = defaultdict(list)
    type_preds: dict = defaultdict(list)
    type_gts: dict = defaultdict(list)
    for item in per_item:
        atype = item["answer_type"]
        type_questions[atype].append(item["question"])
        type_preds[atype].append(item["prediction"])
        type_gts[atype].append(item["ground_truth"])
    type_metrics = {
        t: compute_metrics(accuracy_llm, type_questions[t], type_preds[t], type_gts[t], "QASPER") for t in sorted(type_preds)
    }

    report = {
        "dataset": "QASPER",
        "sample_size": sample_size,
        "concurrency": config.CONCURRENCY,
        "num_papers": len(papers),
        "num_qa_total": len(all_qa),
        "num_retrieve": len(retrieve_qa),
        "num_insert": len(insert_papers),
        "num_delete": len(insert_papers),
        "bulk_insert_time": bulk_time,
        "qa_metrics": qa_metrics,
        "qa_metrics_by_answer_type": type_metrics,
        "insert_metrics": op.summary("insert"),
        "retrieve_metrics": op.summary("retrieve"),
        "delete_metrics": op.summary("delete"),
        "per_item": per_item,
    }

    _print_report(report)
    return report


def _print_report(r: Dict) -> None:
    print("\n" + "-" * 60)
    print("  QASPER Results")
    print("-" * 60)
    qm = r["qa_metrics"]
    print(f"  F1:       {qm['f1']:.4f}")
    print(f"  Recall:   {qm['recall']:.4f}")
    print(f"  Accuracy: {qm['accuracy']:.4f}")
    print(f"  Papers: {r['num_papers']}  Total QA: {r['num_qa_total']}")
    print()
    for t, m in r.get("qa_metrics_by_answer_type", {}).items():
        print(f"  [{t}]  F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Acc={m['accuracy']:.4f}")
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
