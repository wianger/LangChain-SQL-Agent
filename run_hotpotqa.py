"""HotpotQA evaluation pipeline (async-concurrent version).

HotpotQA is a multi-hop QA benchmark requiring reasoning over multiple
Wikipedia paragraphs.  The *HotpotQA_small* subset contains 100 questions
(type: comparison / bridge, level: hard) and 991 articles.

Data files
----------
- hotpot_qa_100.json   : 100 QA items with question, answer, type, level,
                         supporting_facts {title, sent_id}, context {title, sentences}
- hotpot_articles.json : 991 Wikipedia articles with title, text (2-D sentence array)

Pipeline
--------
1.  Load articles -> populate SQLite (articles + article_chunks)
2.  Agent-based INSERT / RETRIEVE / DELETE with asyncio concurrency
3.  Report F1, recall, accuracy + time / token costs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
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
CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    wiki_id     TEXT,
    title       TEXT,
    url         TEXT,
    content     TEXT
);

CREATE TABLE IF NOT EXISTS article_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    wiki_id     TEXT,
    title       TEXT,
    chunk_index INTEGER,
    content     TEXT
);
"""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ── Data loading helpers ─────────────────────────────────────────────────────


def _load_qa() -> List[Dict]:
    with open(config.HOTPOTQA_QA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    qa_list: List[Dict] = []
    for item in raw:
        sf = item.get("supporting_facts", {})
        sf_titles = sf.get("title", [])
        sf_sent_ids = sf.get("sent_id", [])

        ctx = item.get("context", {})
        ctx_titles = ctx.get("title", [])
        ctx_sentences = ctx.get("sentences", [])

        evidence_sentences: List[str] = []
        title_to_sents: Dict[str, List[str]] = {}
        for t, sents in zip(ctx_titles, ctx_sentences):
            title_to_sents[t] = sents

        for t, sid in zip(sf_titles, sf_sent_ids):
            sents = title_to_sents.get(t, [])
            if 0 <= sid < len(sents):
                evidence_sentences.append(sents[sid].strip())

        context_articles: List[Dict[str, str]] = []
        for t, sents in zip(ctx_titles, ctx_sentences):
            full = " ".join(s.strip() for s in sents if s.strip())
            full = re.sub(r"<[^>]+>", "", full)
            if full:
                context_articles.append({"title": t, "content": full})

        qa_list.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item.get("type", ""),
                "level": item.get("level", ""),
                "supporting_titles": list(set(sf_titles)),
                "evidence_sentences": evidence_sentences,
                "context_articles": context_articles,
            }
        )
    return qa_list


def _load_articles() -> List[Dict]:
    with open(config.HOTPOTQA_ARTICLES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    articles: List[Dict] = []
    for item in raw:
        paragraphs: List[str] = []
        for para in item.get("text", []):
            flat_sents: List[str] = []
            for seg in para:
                if isinstance(seg, list):
                    flat_sents.extend(seg)
                else:
                    flat_sents.append(seg)
            paragraph_text = " ".join(s.strip() for s in flat_sents if s.strip())
            if paragraph_text:
                paragraphs.append(paragraph_text)

        content = "\n\n".join(paragraphs)
        content = re.sub(r"<[^>]+>", "", content)

        articles.append(
            {
                "wiki_id": str(item.get("id", "")),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": content,
            }
        )
    return articles


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


def _bulk_insert(db_path: str, articles: List[Dict]) -> float:
    engine = create_engine(f"sqlite:///{db_path}")
    t0 = time.time()
    total_chunks = 0

    with engine.connect() as conn:
        for art in tqdm(articles, desc="loading articles"):
            conn.execute(
                text(
                    "INSERT INTO articles (wiki_id, title, url, content) "
                    "VALUES (:a, :b, :c, :d)"
                ),
                {
                    "a": art["wiki_id"],
                    "b": art["title"],
                    "c": art["url"],
                    "d": art["content"],
                },
            )

            for ci, chunk in enumerate(_chunk_text(art["content"])):
                conn.execute(
                    text(
                        "INSERT INTO article_chunks "
                        "(wiki_id, title, chunk_index, content) "
                        "VALUES (:a, :b, :c, :d)"
                    ),
                    {"a": art["wiki_id"], "b": art["title"], "c": ci, "d": chunk},
                )
                total_chunks += 1

        conn.commit()

    elapsed = time.time() - t0
    engine.dispose()
    logger.info(
        "Bulk insert: %d articles, %d chunks in %.2fs",
        len(articles),
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


def _insert_context_group(engine, qa: Dict) -> None:
    """Insert the context articles embedded in a QA item."""
    qa_id = qa["id"]
    with engine.connect() as conn:
        for art in qa.get("context_articles", []):
            conn.execute(
                text(
                    "INSERT INTO articles (wiki_id, title, url, content) "
                    "VALUES (:a, :b, :c, :d)"
                ),
                {"a": qa_id, "b": art["title"], "c": "", "d": art["content"]},
            )
            for ci, chunk in enumerate(_chunk_text(art["content"])):
                conn.execute(
                    text(
                        "INSERT INTO article_chunks "
                        "(wiki_id, title, chunk_index, content) "
                        "VALUES (:a, :b, :c, :d)"
                    ),
                    {"a": qa_id, "b": art["title"], "c": ci, "d": chunk},
                )
        conn.commit()


def _delete_context_group(engine, qa_id: str) -> None:
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM articles WHERE wiki_id = :q"), {"q": qa_id})
        conn.execute(text("DELETE FROM article_chunks WHERE wiki_id = :q"), {"q": qa_id})
        conn.commit()


async def run_experiment(
    sample_size: int = config.SMALL_SAMPLE_SIZE,
    verbose: bool = False,
    mode: str = "bulk",
) -> Dict[str, Any]:

    print("\n" + "=" * 60)
    print(
        "  HotpotQA Evaluation  (sample_size=%d, concurrency=%d, mode=%s)"
        % (sample_size, config.CONCURRENCY, mode)
    )
    print("=" * 60)

    all_qa = _load_qa()
    articles = _load_articles()

    db_path = config.HOTPOTQA_DB
    _init_db(db_path)

    if mode == "per-question":
        tracker = TokenTracker(config.TIKTOKEN_ENCODING)
        op = OperationTracker(tracker)
        agent = build_agent(db_path, tracker, verbose=verbose)
        accuracy_llm = get_llm()
        sem = asyncio.Semaphore(config.CONCURRENCY)

        rng = random.Random(42)
        retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))

        async def _pq_retrieve(qa):
            title_hints = ", ".join(f"'{t}'" for t in qa["supporting_titles"][:3])
            prompt = (
                f"Using the Wikipedia articles stored in the database, "
                f"answer the following multi-hop question. "
                f"Hint: relevant articles may include {title_hints}. "
                f"Search in both 'articles' and 'article_chunks' tables. "
                f"Give a concise answer. "
                f"If the answer cannot be found, reply 'unanswerable'.\n\n"
                f"Question: {qa['question']}"
            )
            async with sem:
                with op.track("retrieve"):
                    return await arun_agent(agent, prompt)

        engine = create_engine(f"sqlite:///{db_path}")
        setup_time = 0.0
        teardown_time = 0.0
        qa_answers: list[tuple] = []

        pbar = tqdm(total=len(retrieve_qa), desc="per-question retrieve")
        for qa in retrieve_qa:
            t0 = time.time()
            _insert_context_group(engine, qa)
            setup_time += time.time() - t0

            ans = await _pq_retrieve(qa)
            qa_answers.append((qa, ans))
            pbar.update(1)

            t0 = time.time()
            _delete_context_group(engine, qa["id"])
            teardown_time += time.time() - t0
        pbar.close()
        engine.dispose()

        questions: list[str] = []
        predictions: list[str] = []
        ground_truths: list[Union[str, List[str]]] = []
        per_item: list[dict] = []
        for qa, answer in qa_answers:
            answer = answer or ""
            predictions.append(answer)
            questions.append(qa["question"])
            gt = qa["answer"]
            ground_truths.append(gt)
            per_item.append(
                {
                    "id": qa["id"],
                    "question": qa["question"],
                    "type": qa["type"],
                    "level": qa["level"],
                    "ground_truth": gt,
                    "evidence_sentences": qa["evidence_sentences"],
                    "prediction": answer,
                    "f1": token_f1(answer, gt),
                    "recall": token_recall(answer, gt),
                    "accuracy": accuracy(
                        accuracy_llm, qa["question"], answer, gt, "hotpotqa"
                    ),
                }
            )

        qa_metrics = compute_metrics(
            accuracy_llm, questions, predictions, ground_truths, "hotpotqa"
        )
        type_preds: dict = defaultdict(list)
        type_gts: dict = defaultdict(list)
        type_questions: dict = defaultdict(list)
        for item in per_item:
            t = item["type"]
            type_preds[t].append(item["prediction"])
            type_gts[t].append(item["ground_truth"])
            type_questions[t].append(item["question"])
        type_metrics = {
            t: compute_metrics(
                accuracy_llm, type_questions[t], type_preds[t], type_gts[t], "hotpotqa"
            )
            for t in sorted(type_preds)
        }

        report = {
            "dataset": "HotpotQA",
            "mode": "per-question",
            "sample_size": sample_size,
            "concurrency": config.CONCURRENCY,
            "num_articles": len(articles),
            "num_qa_total": len(all_qa),
            "num_doc_groups": len(retrieve_qa),
            "num_retrieve": len(retrieve_qa),
            "setup_time": setup_time,
            "teardown_time": teardown_time,
            "qa_metrics": qa_metrics,
            "qa_metrics_by_type": type_metrics,
            "retrieve_metrics": op.summary("retrieve"),
            "per_item": per_item,
        }
        _print_report(report)
        return report

    bulk_time = _bulk_insert(db_path, articles)

    tracker = TokenTracker(config.TIKTOKEN_ENCODING)
    op = OperationTracker(tracker)
    agent = build_agent(db_path, tracker, verbose=verbose)
    accuracy_llm = get_llm()

    rng = random.Random(42)
    retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))

    wiki_ids = list({a["wiki_id"] for a in articles})
    insert_ids = rng.sample(wiki_ids, min(sample_size, len(wiki_ids)))

    sem = asyncio.Semaphore(config.CONCURRENCY)

    # ── INSERT ──────────────────────────────────────────────────────────────
    print(
        "\n[INSERT] %d records (concurrency=%d)..."
        % (len(insert_ids), config.CONCURRENCY)
    )

    async def _do_insert(wiki_id):
        prompt = (
            f"Insert a new record into the article_chunks table with: "
            f"wiki_id='{wiki_id}', title='test', chunk_index=9999, "
            f"content='[test insert placeholder for {wiki_id}]'"
        )
        async with sem:
            with op.track("insert"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_insert(wid) for wid in insert_ids], "insert")

    # ── RETRIEVE ────────────────────────────────────────────────────────────
    print(
        "\n[RETRIEVE] %d questions (concurrency=%d)..."
        % (len(retrieve_qa), config.CONCURRENCY)
    )

    async def _do_retrieve(qa):
        title_hints = ", ".join(f"'{t}'" for t in qa["supporting_titles"][:3])
        prompt = (
            f"Using the Wikipedia articles stored in the database, "
            f"answer the following multi-hop question. "
            f"Hint: relevant articles may include {title_hints}. "
            f"Search in both 'articles' and 'article_chunks' tables. "
            f"Give a concise answer. "
            f"If the answer cannot be found, reply 'unanswerable'.\n\n"
            f"Question: {qa['question']}"
        )
        async with sem:
            with op.track("retrieve"):
                return await arun_agent(agent, prompt)

    answers = await _run_concurrent(
        [_do_retrieve(qa) for qa in retrieve_qa], "retrieve"
    )

    questions: list[str] = []
    predictions: list[str] = []
    ground_truths: list[Union[str, List[str]]] = []
    per_item: list[dict] = []
    for qa, answer in zip(retrieve_qa, answers):
        answer = answer or ""
        predictions.append(answer)
        questions.append(qa["question"])
        gt = qa["answer"]
        ground_truths.append(gt)
        per_item.append(
            {
                "id": qa["id"],
                "question": qa["question"],
                "type": qa["type"],
                "level": qa["level"],
                "ground_truth": gt,
                "evidence_sentences": qa["evidence_sentences"],
                "prediction": answer,
                "f1": token_f1(answer, gt),
                "recall": token_recall(answer, gt),
                "accuracy": accuracy(
                    accuracy_llm, qa["question"], answer, gt, "hotpotqa"
                ),
            }
        )

    # ── DELETE ──────────────────────────────────────────────────────────────
    print(
        "\n[DELETE] %d records (concurrency=%d)..."
        % (len(insert_ids), config.CONCURRENCY)
    )

    async def _do_delete(wiki_id):
        prompt = (
            f"Delete the record from article_chunks where "
            f"wiki_id = '{wiki_id}' AND chunk_index = 9999."
        )
        async with sem:
            with op.track("delete"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_delete(wid) for wid in insert_ids], "delete")

    # ── Metrics ─────────────────────────────────────────────────────────────
    qa_metrics = compute_metrics(
        accuracy_llm, questions, predictions, ground_truths, "hotpotqa"
    )

    type_preds: dict = defaultdict(list)
    type_gts: dict = defaultdict(list)
    type_questions: dict = defaultdict(list)
    for item in per_item:
        t = item["type"]
        type_preds[t].append(item["prediction"])
        type_gts[t].append(item["ground_truth"])
        type_questions[t].append(item["question"])
    type_metrics = {
        t: compute_metrics(
            accuracy_llm, type_questions[t], type_preds[t], type_gts[t], "hotpotqa"
        )
        for t in sorted(type_preds)
    }

    report = {
        "dataset": "HotpotQA",
        "sample_size": sample_size,
        "concurrency": config.CONCURRENCY,
        "num_articles": len(articles),
        "num_qa_total": len(all_qa),
        "num_retrieve": len(retrieve_qa),
        "num_insert": len(insert_ids),
        "num_delete": len(insert_ids),
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
    print(f"  HotpotQA Results  (mode={r.get('mode', 'bulk')})")
    print("-" * 60)
    qm = r["qa_metrics"]
    print(f"  F1:       {qm['f1']:.4f}")
    print(f"  Recall:   {qm['recall']:.4f}")
    print(f"  Accuracy: {qm['accuracy']:.4f}")
    print(f"  Articles: {r['num_articles']}  Total QA: {r['num_qa_total']}")
    print()
    for t, m in r.get("qa_metrics_by_type", {}).items():
        print(
            f"  [{t}]  F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Acc={m['accuracy']:.4f}"
        )
    print()
    if "bulk_insert_time" in r:
        print(f"  Bulk insert time: {r['bulk_insert_time']:.2f}s")
    if "setup_time" in r:
        print(
            f"  Doc groups: {r['num_doc_groups']}  "
            f"Setup: {r['setup_time']:.2f}s  Teardown: {r['teardown_time']:.2f}s"
        )

    for op_name in ("insert", "retrieve", "delete"):
        m = r.get(f"{op_name}_metrics", {})
        if m:
            print(
                f"  {op_name:>8}: n={m['count']}  "
                f"avg_time={m['avg_time']:.2f}s  total_time={m['total_time']:.2f}s  "
                f"avg_tokens={m['avg_tokens']:.0f}  total_tokens={m['total_tokens']}"
            )
    print("-" * 60)
