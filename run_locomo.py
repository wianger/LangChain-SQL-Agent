"""LoCoMo evaluation pipeline (async-concurrent version).

1.  Load locomo10.json -> populate SQLite
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
from typing import Any, Dict, List, Tuple

from sqlalchemy import create_engine, text
from tqdm import tqdm

import config
from metrics import accuracy, compute_metrics, token_f1, token_recall
from sql_agent import arun_agent, build_agent, get_llm
from token_tracker import OperationTracker, TokenTracker

logger = logging.getLogger(__name__)

# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id   TEXT,
    session_id  INTEGER,
    session_date TEXT,
    turn_number INTEGER,
    dia_id      TEXT,
    speaker     TEXT,
    text        TEXT
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id   TEXT,
    session_id  INTEGER,
    summary     TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id   TEXT,
    session_id  INTEGER,
    speaker     TEXT,
    observation TEXT,
    dia_id      TEXT
);
"""


# ── Data loading helpers ─────────────────────────────────────────────────────


def _load_json() -> List[Dict]:
    with open(config.LOCOMO_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_rows(data: List[Dict]) -> Tuple[list, list, list]:
    conv_rows: list = []
    summary_rows: list = []
    obs_rows: list = []

    for entry in data:
        sid = entry["sample_id"]
        conv = entry["conversation"]

        for i in range(1, 100):
            sess_key = f"session_{i}"
            date_key = f"{sess_key}_date_time"
            if sess_key not in conv or not isinstance(conv[sess_key], list):
                continue
            sess_date = conv.get(date_key, "")
            for turn_idx, turn in enumerate(conv[sess_key]):
                conv_rows.append(
                    (
                        sid,
                        i,
                        sess_date,
                        turn_idx + 1,
                        turn.get("dia_id", ""),
                        turn.get("speaker", ""),
                        turn.get("text", ""),
                    )
                )

        for i in range(1, 100):
            skey = f"session_{i}_summary"
            if skey in entry.get("session_summary", {}):
                summary_rows.append((sid, i, entry["session_summary"][skey]))

        for i in range(1, 100):
            okey = f"session_{i}_observation"
            obs_dict = entry.get("observation", {}).get(okey)
            if not obs_dict or not isinstance(obs_dict, dict):
                continue
            for speaker, obs_list in obs_dict.items():
                if not isinstance(obs_list, list):
                    continue
                for item in obs_list:
                    if isinstance(item, list) and len(item) >= 2:
                        dia_ref = item[1]
                        if isinstance(dia_ref, list):
                            dia_ref = ", ".join(str(x) for x in dia_ref)
                        obs_rows.append((sid, i, speaker, str(item[0]), str(dia_ref)))

    return conv_rows, summary_rows, obs_rows


def _extract_qa(data: List[Dict]) -> List[Dict]:
    qa_list: list = []
    for entry in data:
        sid = entry["sample_id"]
        speakers = (
            entry["conversation"].get("speaker_a", ""),
            entry["conversation"].get("speaker_b", ""),
        )
        for qa in entry["qa"]:
            cat = qa["category"]
            question = qa["question"]
            if cat == 5:
                answer = "unanswerable"
            else:
                answer = str(qa.get("answer", ""))
            qa_list.append(
                {
                    "sample_id": sid,
                    "question": question,
                    "answer": answer,
                    "category": cat,
                    "speakers": speakers,
                }
            )
    return qa_list


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


def _bulk_insert(db_path: str, data: List[Dict]) -> float:
    conv_rows, summary_rows, obs_rows = _extract_rows(data)
    engine = create_engine(f"sqlite:///{db_path}")
    t0 = time.time()
    with engine.connect() as conn:
        for r in conv_rows:
            conn.execute(
                text(
                    "INSERT INTO conversations "
                    "(sample_id, session_id, session_date, turn_number, dia_id, speaker, text) "
                    "VALUES (:a,:b,:c,:d,:e,:f,:g)"
                ),
                {
                    "a": r[0],
                    "b": r[1],
                    "c": r[2],
                    "d": r[3],
                    "e": r[4],
                    "f": r[5],
                    "g": r[6],
                },
            )
        for r in summary_rows:
            conn.execute(
                text(
                    "INSERT INTO session_summaries (sample_id, session_id, summary) "
                    "VALUES (:a,:b,:c)"
                ),
                {"a": r[0], "b": r[1], "c": r[2]},
            )
        for r in obs_rows:
            conn.execute(
                text(
                    "INSERT INTO observations "
                    "(sample_id, session_id, speaker, observation, dia_id) "
                    "VALUES (:a,:b,:c,:d,:e)"
                ),
                {"a": r[0], "b": r[1], "c": r[2], "d": r[3], "e": r[4]},
            )
        conn.commit()
    elapsed = time.time() - t0
    engine.dispose()
    logger.info(
        "Bulk insert: %d conv + %d summaries + %d obs in %.2fs",
        len(conv_rows),
        len(summary_rows),
        len(obs_rows),
        elapsed,
    )
    return elapsed


# ── Async helpers ────────────────────────────────────────────────────────────


async def _run_concurrent(coros: list, desc: str) -> list:
    """Run coroutines concurrently with a tqdm progress bar."""
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
        "  LoCoMo Evaluation  (sample_size=%d, concurrency=%d)"
        % (sample_size, config.CONCURRENCY)
    )
    print("=" * 60)

    data = _load_json()
    all_qa = _extract_qa(data)
    qa_with_answer = [q for q in all_qa if q["category"] != 5]
    qa_adversarial = [q for q in all_qa if q["category"] == 5]

    db_path = config.LOCOMO_DB
    _init_db(db_path)
    bulk_time = _bulk_insert(db_path, data)

    tracker = TokenTracker(config.TIKTOKEN_ENCODING)
    op = OperationTracker(tracker)
    agent = build_agent(db_path, tracker, verbose=verbose)

    rng = random.Random(42)
    retrieve_qa = rng.sample(qa_with_answer, min(sample_size, len(qa_with_answer)))
    if qa_adversarial:
        adv_sample = rng.sample(
            qa_adversarial, min(max(sample_size // 5, 2), len(qa_adversarial))
        )
        retrieve_qa.extend(adv_sample)

    conv_rows, _, _ = _extract_rows(data)
    insert_sample = rng.sample(conv_rows, min(sample_size, len(conv_rows)))
    delete_sample_ids = [r[4] for r in insert_sample[:sample_size]]

    sem = asyncio.Semaphore(config.CONCURRENCY)

    # ── INSERT ──────────────────────────────────────────────────────────────
    print(
        "\n[INSERT] %d records (concurrency=%d)..."
        % (len(insert_sample), config.CONCURRENCY)
    )

    async def _do_insert(row):
        prompt = (
            f"Insert a new record into the conversations table with: "
            f"sample_id='{row[0]}', session_id={row[1]}, "
            f"session_date='{row[2]}', turn_number={row[3]}, "
            f"dia_id='{row[4]}_dup', speaker='{row[5]}', "
            f"text='{row[6][:200]}'"
        )
        async with sem:
            with op.track("insert"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_insert(r) for r in insert_sample], "insert")

    # ── RETRIEVE ────────────────────────────────────────────────────────────
    print(
        "\n[RETRIEVE] %d questions (concurrency=%d)..."
        % (len(retrieve_qa), config.CONCURRENCY)
    )

    async def _do_retrieve(qa):
        prompt = (
            f"Based on conversations with sample_id '{qa['sample_id']}' "
            f"between {qa['speakers'][0]} and {qa['speakers'][1]}, "
            f"answer the following question. If the answer cannot be found, "
            f"reply 'unanswerable'.\n\nQuestion: {qa['question']}"
        )
        async with sem:
            with op.track("retrieve"):
                return await arun_agent(agent, prompt)

    answers = await _run_concurrent(
        [_do_retrieve(qa) for qa in retrieve_qa], "retrieve"
    )

    accuracy_llm = get_llm()
    predictions: list[str] = []
    ground_truths: list[str] = []
    per_item: list[dict] = []
    for qa, answer in zip(retrieve_qa, answers):
        answer = answer or ""
        predictions.append(answer)
        ground_truths.append(qa["answer"])
        per_item.append(
            {
                "question": qa["question"],
                "ground_truth": qa["answer"],
                "prediction": answer,
                "category": qa["category"],
                "f1": token_f1(answer, qa["answer"]),
                "recall": token_recall(answer, qa["answer"]),
                "accuracy": accuracy(accuracy_llm, answer, qa["answer"], "Locomo"),
            }
        )

    # ── DELETE ──────────────────────────────────────────────────────────────
    print(
        "\n[DELETE] %d records (concurrency=%d)..."
        % (len(delete_sample_ids), config.CONCURRENCY)
    )

    async def _do_delete(dia_id):
        prompt = f"Delete the record from conversations where dia_id = '{dia_id}_dup'."
        async with sem:
            with op.track("delete"):
                await arun_agent(agent, prompt)

    await _run_concurrent([_do_delete(d) for d in delete_sample_ids], "delete")

    # ── Metrics ─────────────────────────────────────────────────────────────
    qa_metrics = compute_metrics(accuracy_llm, predictions, ground_truths, "Locomo")

    cat_preds: dict = defaultdict(list)
    cat_gts: dict = defaultdict(list)
    for item in per_item:
        cat_preds[item["category"]].append(item["prediction"])
        cat_gts[item["category"]].append(item["ground_truth"])
    cat_metrics = {
        config.LOCOMO_CATEGORIES.get(cat, str(cat)): compute_metrics(
            accuracy_llm, cat_preds[cat], cat_gts[cat], "Locomo"
        )
        for cat in sorted(cat_preds)
    }

    report = {
        "dataset": "LoCoMo",
        "sample_size": sample_size,
        "concurrency": config.CONCURRENCY,
        "num_retrieve": len(retrieve_qa),
        "num_insert": len(insert_sample),
        "num_delete": len(delete_sample_ids),
        "bulk_insert_time": bulk_time,
        "qa_metrics": qa_metrics,
        "qa_metrics_by_category": cat_metrics,
        "insert_metrics": op.summary("insert"),
        "retrieve_metrics": op.summary("retrieve"),
        "delete_metrics": op.summary("delete"),
        "per_item": per_item,
    }

    _print_report(report)
    return report


def _print_report(r: Dict) -> None:
    print("\n" + "-" * 60)
    print("  LoCoMo Results")
    print("-" * 60)
    qm = r["qa_metrics"]
    print(f"  F1:       {qm['f1']:.4f}")
    print(f"  Recall:   {qm['recall']:.4f}")
    print(f"  Accuracy: {qm['accuracy']:.4f}")
    print()
    for cat, m in r["qa_metrics_by_category"].items():
        print(
            f"  [{cat}]  F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Acc={m['accuracy']:.4f}"
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
