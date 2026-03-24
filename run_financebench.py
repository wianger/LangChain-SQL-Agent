"""FinanceBench evaluation pipeline (async-concurrent version).

1.  Load financebench_open_source.jsonl + extract text from PDFs -> populate SQLite
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
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_name    TEXT UNIQUE,
    company     TEXT,
    gics_sector TEXT,
    doc_type    TEXT,
    doc_period  INTEGER,
    num_pages   INTEGER,
    content     TEXT
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_name    TEXT,
    chunk_index INTEGER,
    page_num    INTEGER,
    content     TEXT
);
"""

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ── PDF text extraction ──────────────────────────────────────────────────────


def _is_docling_available() -> bool:
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file using priority: docling → pypdf → pdfplumber."""

    # 1) docling — highest quality, preserves table structure as Markdown
    if _is_docling_available():
        try:
            from docling.document_converter import DocumentConverter  # type: ignore

            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            content = result.document.export_to_markdown()
            if content.strip():
                logger.info("PDF extracted via docling: %s", pdf_path)
                return content
        except Exception as exc:
            logger.warning("docling failed for %s: %s, falling back", pdf_path, exc)

    # 2) pypdf — default fallback
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            reader.decrypt("")
        content = ""
        for page in reader.pages:
            content += (page.extract_text() or "") + "\n"
        if content.strip():
            logger.info("PDF extracted via pypdf: %s", pdf_path)
            return content
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pypdf failed for %s: %s, falling back", pdf_path, exc)

    # 3) pdfplumber — additional fallback
    try:
        import pdfplumber

        pages_text: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
        content = "\n\n".join(pages_text)
        if content.strip():
            logger.info("PDF extracted via pdfplumber: %s", pdf_path)
            return content
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s", pdf_path, exc)

    raise RuntimeError(
        f"Cannot extract text from {pdf_path}. "
        "Install one of: pip install 'docling>=2' / pip install pypdf / pip install pdfplumber"
    )


def _chunk_text(text_content: str) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks."""
    if not text_content:
        return []
    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(text_content):
        end = start + CHUNK_SIZE
        chunks.append(
            {
                "chunk_index": idx,
                "page_num": 0,
                "content": text_content[start:end],
            }
        )
        idx += 1
        start = end - CHUNK_OVERLAP
    return chunks


# ── Data loading helpers ─────────────────────────────────────────────────────


def _load_qa() -> List[Dict]:
    items: List[Dict] = []
    with open(config.FINANCEBENCH_QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_doc_info() -> Dict[str, Dict]:
    info: Dict[str, Dict] = {}
    with open(config.FINANCEBENCH_DOC_INFO_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                info[row["doc_name"]] = row
    return info


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
    """Extract text from referenced PDFs and insert into SQLite."""
    doc_info = _load_doc_info()
    needed_docs = sorted({q["doc_name"] for q in qa_data})

    engine = create_engine(f"sqlite:///{db_path}")
    t0 = time.time()

    total_chunks = 0
    with engine.connect() as conn:
        for doc_name in tqdm(needed_docs, desc="loading PDFs"):
            pdf_path = os.path.join(config.FINANCEBENCH_PDF_DIR, doc_name + ".pdf")
            try:
                full_text = _extract_pdf_text(pdf_path)
            except RuntimeError as exc:
                logger.error(str(exc))
                full_text = ""

            meta = doc_info.get(doc_name, {})

            conn.execute(
                text(
                    "INSERT OR IGNORE INTO documents "
                    "(doc_name, company, gics_sector, doc_type, doc_period, num_pages, content) "
                    "VALUES (:a,:b,:c,:d,:e,:f,:g)"
                ),
                {
                    "a": doc_name,
                    "b": meta.get("company", ""),
                    "c": meta.get("gics_sector", ""),
                    "d": meta.get("doc_type", ""),
                    "e": meta.get("doc_period", 0),
                    "f": 0,
                    "g": full_text,
                },
            )

            for chunk in _chunk_text(full_text):
                conn.execute(
                    text(
                        "INSERT INTO document_chunks "
                        "(doc_name, chunk_index, page_num, content) "
                        "VALUES (:a,:b,:c,:d)"
                    ),
                    {
                        "a": doc_name,
                        "b": chunk["chunk_index"],
                        "c": chunk["page_num"],
                        "d": chunk["content"],
                    },
                )
                total_chunks += 1

        conn.commit()

    elapsed = time.time() - t0
    engine.dispose()
    logger.info(
        "Bulk insert: %d documents, %d chunks in %.2fs",
        len(needed_docs),
        total_chunks,
        elapsed,
    )
    return elapsed


def _bulk_delete(db_path: str) -> float:
    """Delete all documents and chunks from SQLite."""
    engine = create_engine(f"sqlite:///{db_path}")
    t0 = time.time()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM documents"))
        conn.execute(text("DELETE FROM document_chunks"))
        conn.commit()
    elapsed = time.time() - t0
    engine.dispose()
    logger.info("Bulk delete: all documents and chunks deleted in %.2fs", elapsed)
    return elapsed


def _insert_doc_group(engine, doc_name: str, doc_info: Dict, pdf_dir: str) -> None:
    """Insert one document + chunks for per-question mode."""
    pdf_path = os.path.join(pdf_dir, doc_name + ".pdf")
    try:
        full_text = _extract_pdf_text(pdf_path)
    except RuntimeError:
        full_text = ""
    meta = doc_info.get(doc_name, {})
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT OR IGNORE INTO documents "
                "(doc_name, company, gics_sector, doc_type, doc_period, num_pages, content) "
                "VALUES (:a,:b,:c,:d,:e,:f,:g)"
            ),
            {
                "a": doc_name,
                "b": meta.get("company", ""),
                "c": meta.get("gics_sector", ""),
                "d": meta.get("doc_type", ""),
                "e": meta.get("doc_period", 0),
                "f": 0,
                "g": full_text,
            },
        )
        for chunk in _chunk_text(full_text):
            conn.execute(
                text(
                    "INSERT INTO document_chunks "
                    "(doc_name, chunk_index, page_num, content) "
                    "VALUES (:a,:b,:c,:d)"
                ),
                {
                    "a": doc_name,
                    "b": chunk["chunk_index"],
                    "c": chunk["page_num"],
                    "d": chunk["content"],
                },
            )
        conn.commit()


def _delete_doc_group(engine, doc_name: str) -> None:
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM documents WHERE doc_name = :d"), {"d": doc_name})
        conn.execute(
            text("DELETE FROM document_chunks WHERE doc_name = :d"), {"d": doc_name}
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
        "  FinanceBench Evaluation  (sample_size=%d, concurrency=%d, mode=%s)"
        % (sample_size, config.CONCURRENCY, mode)
    )
    print("=" * 60)

    all_qa = _load_qa()

    db_path = config.FINANCEBENCH_DB
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
        doc_info = _load_doc_info()

        qa_by_doc: Dict[str, List[Dict]] = defaultdict(list)
        for qa in retrieve_qa:
            qa_by_doc[qa["doc_name"]].append(qa)

        setup_time = 0.0
        teardown_time = 0.0
        answers_by_qa: Dict[tuple, tuple] = {}

        for doc_name, qa_list in qa_by_doc.items():
            t0 = time.time()
            _insert_doc_group(engine, doc_name, doc_info, config.FINANCEBENCH_PDF_DIR)
            setup_time += time.time() - t0

            async def _do_retrieve(qa):
                prompt = (
                    f"Using the financial document data for '{qa['doc_name']}' "
                    f"(company: {qa['company']}) stored in the database, "
                    f"answer the following question. "
                    f"Search in both 'documents' and 'document_chunks' tables. "
                    f"If the answer cannot be found, reply 'unanswerable'.\n\n"
                    f"Question: {qa['question']}"
                )
                async with sem:
                    with op.track("retrieve"):
                        return await arun_agent(agent, prompt)

            results = await _run_concurrent(
                [_do_retrieve(qa) for qa in qa_list], f"retrieve {doc_name}"
            )
            for qa, (answer, retrieved) in zip(qa_list, results):
                key = (qa["financebench_id"], qa["question"])
                answers_by_qa[key] = (answer or "", retrieved or [])

            t1 = time.time()
            _delete_doc_group(engine, doc_name)
            teardown_time += time.time() - t1

        engine.dispose()

        per_item: list = []
        all_retrieved: list[list[str]] = []
        evidences: list[list[str]] = []
        for qa in retrieve_qa:
            key = (qa["financebench_id"], qa["question"])
            answer, retrieved = answers_by_qa.get(key, ("", []))
            gt = qa["answer"]
            ev = [
                e["evidence_text"]
                for e in qa.get("evidence", [])
                if e.get("evidence_text")
            ]
            if not ev:
                ev = [gt]
            all_retrieved.append(retrieved)
            evidences.append(ev)
            per_item.append(
                {
                    "financebench_id": qa["financebench_id"],
                    "question": qa["question"],
                    "doc_name": qa["doc_name"],
                    "company": qa["company"],
                    "question_type": qa.get("question_type", ""),
                    "question_reasoning": qa.get("question_reasoning") or "unknown",
                    "ground_truth": gt,
                    "evidence_texts": ev,
                    "prediction": answer,
                    "retrieved_texts": retrieved,
                    "f1": token_f1(answer, gt),
                    "recall": token_recall(answer, ev, retrieved_texts=retrieved),
                    "accuracy": accuracy(
                        accuracy_llm, qa["question"], answer, gt, "financebench"
                    ),
                }
            )

        predictions = [p["prediction"] for p in per_item]
        ground_truths = [p["ground_truth"] for p in per_item]
        questions = [p["question"] for p in per_item]

        qa_metrics = compute_metrics(
            accuracy_llm,
            questions,
            predictions,
            ground_truths,
            "financebench",
            evidences=evidences,
            retrieved_texts_list=all_retrieved,
        )

        type_preds: dict = defaultdict(list)
        type_gts: dict = defaultdict(list)
        type_questions: dict = defaultdict(list)
        type_evidences: dict = defaultdict(list)
        type_retrieved: dict = defaultdict(list)
        for item in per_item:
            qtype = item["question_type"]
            type_questions[qtype].append(item["question"])
            type_preds[qtype].append(item["prediction"])
            type_gts[qtype].append(item["ground_truth"])
            type_evidences[qtype].append(item["evidence_texts"])
            type_retrieved[qtype].append(item["retrieved_texts"])
        type_metrics = {
            t: compute_metrics(
                accuracy_llm,
                type_questions[t],
                type_preds[t],
                type_gts[t],
                "financebench",
                evidences=type_evidences[t],
                retrieved_texts_list=type_retrieved[t],
            )
            for t in sorted(type_preds)
        }

        reasoning_preds: dict = defaultdict(list)
        reasoning_gts: dict = defaultdict(list)
        reasoning_questions: dict = defaultdict(list)
        reasoning_evidences: dict = defaultdict(list)
        reasoning_retrieved: dict = defaultdict(list)
        for item in per_item:
            r = item["question_reasoning"]
            reasoning_preds[r].append(item["prediction"])
            reasoning_gts[r].append(item["ground_truth"])
            reasoning_questions[r].append(item["question"])
            reasoning_evidences[r].append(item["evidence_texts"])
            reasoning_retrieved[r].append(item["retrieved_texts"])
        reasoning_metrics = {
            r: compute_metrics(
                accuracy_llm,
                reasoning_questions[r],
                reasoning_preds[r],
                reasoning_gts[r],
                "financebench",
                evidences=reasoning_evidences[r],
                retrieved_texts_list=reasoning_retrieved[r],
            )
            for r in sorted(reasoning_preds)
        }

        report = {
            "dataset": "FinanceBench",
            "sample_size": sample_size,
            "concurrency": config.CONCURRENCY,
            "mode": "per-question",
            "num_retrieve": len(retrieve_qa),
            "num_doc_groups": len(qa_by_doc),
            "setup_time": setup_time,
            "teardown_time": teardown_time,
            "qa_metrics": qa_metrics,
            "qa_metrics_by_type": type_metrics,
            "qa_metrics_by_reasoning": reasoning_metrics,
            "retrieve_metrics": op.summary("retrieve"),
            "per_item": per_item,
        }
        _print_report(report)
        return report

    bulk_time = _bulk_insert(db_path, all_qa)

    tracker = TokenTracker(config.TIKTOKEN_ENCODING)
    op = OperationTracker(tracker)
    agent = build_agent(db_path, tracker, verbose=verbose)
    accuracy_llm = get_llm()

    rng = random.Random(42)
    retrieve_qa = rng.sample(all_qa, min(sample_size, len(all_qa)))
    sem = asyncio.Semaphore(config.CONCURRENCY)

    # ── RETRIEVE ────────────────────────────────────────────────────────────
    print(
        "\n[RETRIEVE] %d questions (concurrency=%d)..."
        % (len(retrieve_qa), config.CONCURRENCY)
    )

    async def _do_retrieve(qa):
        prompt = (
            f"Using the financial document data for '{qa['doc_name']}' "
            f"(company: {qa['company']}) stored in the database, "
            f"answer the following question. "
            f"Search in both 'documents' and 'document_chunks' tables. "
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
    ground_truths: list[str] = []
    evidences: list[list[str]] = []
    all_retrieved: list[list[str]] = []
    per_item: list[dict] = []
    for qa, (answer, retrieved) in zip(retrieve_qa, results):
        answer = answer or ""
        retrieved = retrieved or []
        predictions.append(answer)
        questions.append(qa["question"])
        gt = qa["answer"]
        ev = [
            e["evidence_text"] for e in qa.get("evidence", []) if e.get("evidence_text")
        ]
        if not ev:
            ev = [gt]
        ground_truths.append(gt)
        evidences.append(ev)
        all_retrieved.append(retrieved)
        print(f"retrieved: {retrieved}")
        per_item.append(
            {
                "financebench_id": qa["financebench_id"],
                "question": qa["question"],
                "doc_name": qa["doc_name"],
                "company": qa["company"],
                "question_type": qa.get("question_type", ""),
                "question_reasoning": qa.get("question_reasoning") or "unknown",
                "ground_truth": gt,
                "evidence_texts": ev,
                "prediction": answer,
                "retrieved_texts": retrieved,
                "f1": token_f1(answer, gt),
                "recall": token_recall(answer, ev, retrieved_texts=retrieved),
                "accuracy": accuracy(
                    accuracy_llm, qa["question"], answer, gt, "financebench"
                ),
            }
        )

    # ── DELETE ──────────────────────────────────────────────────────────────
    builk_delete_time = _bulk_delete(db_path)

    # ── Metrics ─────────────────────────────────────────────────────────────
    questions = [q["question"] for q in retrieve_qa]
    qa_metrics = compute_metrics(
        accuracy_llm,
        questions,
        predictions,
        ground_truths,
        "financebench",
        evidences=evidences,
        retrieved_texts_list=all_retrieved,
    )

    type_preds: dict = defaultdict(list)
    type_gts: dict = defaultdict(list)
    type_questions: dict = defaultdict(list)
    type_evidences: dict = defaultdict(list)
    type_retrieved: dict = defaultdict(list)
    for item in per_item:
        qtype = item["question_type"]
        type_questions[qtype].append(item["question"])
        type_preds[qtype].append(item["prediction"])
        type_gts[qtype].append(item["ground_truth"])
        type_evidences[qtype].append(item["evidence_texts"])
        type_retrieved[qtype].append(item["retrieved_texts"])
    type_metrics = {
        t: compute_metrics(
            accuracy_llm,
            type_questions[t],
            type_preds[t],
            type_gts[t],
            "financebench",
            evidences=type_evidences[t],
            retrieved_texts_list=type_retrieved[t],
        )
        for t in sorted(type_preds)
    }

    reasoning_preds: dict = defaultdict(list)
    reasoning_gts: dict = defaultdict(list)
    reasoning_questions: dict = defaultdict(list)
    reasoning_evidences: dict = defaultdict(list)
    reasoning_retrieved: dict = defaultdict(list)
    for item in per_item:
        r = item["question_reasoning"]
        reasoning_preds[r].append(item["prediction"])
        reasoning_gts[r].append(item["ground_truth"])
        reasoning_questions[r].append(item["question"])
        reasoning_evidences[r].append(item["evidence_texts"])
        reasoning_retrieved[r].append(item["retrieved_texts"])
    reasoning_metrics = {
        r: compute_metrics(
            accuracy_llm,
            reasoning_questions[r],
            reasoning_preds[r],
            reasoning_gts[r],
            "financebench",
            evidences=reasoning_evidences[r],
            retrieved_texts_list=reasoning_retrieved[r],
        )
        for r in sorted(reasoning_preds)
    }

    report = {
        "dataset": "FinanceBench",
        "sample_size": sample_size,
        "concurrency": config.CONCURRENCY,
        "num_retrieve": len(retrieve_qa),
        "bulk_insert_time": bulk_time,
        "bulk_delete_time": builk_delete_time,
        "qa_metrics": qa_metrics,
        "qa_metrics_by_type": type_metrics,
        "qa_metrics_by_reasoning": reasoning_metrics,
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
    print(f"  FinanceBench Results  (mode={mode})")
    print("-" * 60)
    qm = r["qa_metrics"]
    print(f"  F1:       {qm['f1']:.4f}")
    print(f"  Recall:   {qm['recall']:.4f}")
    print(f"  Accuracy: {qm['accuracy']:.4f}")
    print()
    for t, m in r.get("qa_metrics_by_type", {}).items():
        print(
            f"  [type: {t}]  F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Acc={m['accuracy']:.4f}"
        )
    print()
    for t, m in r.get("qa_metrics_by_reasoning", {}).items():
        print(
            f"  [reasoning: {t}]  F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Acc={m['accuracy']:.4f}"
        )
    print()
    if mode == "bulk":
        print(f"  Bulk insert time: {r['bulk_insert_time']:.2f}s")
        print(f"  Bulk delete time: {r['bulk_delete_time']:.2f}s")
    else:
        print(f"  Setup time: {r['setup_time']:.2f}s")
        print(f"  Teardown time: {r['teardown_time']:.2f}s")
        print(f"  Num doc groups: {r['num_doc_groups']}")

    for op_name in ("insert", "retrieve", "delete"):
        m = r.get(f"{op_name}_metrics", {})
        if m:
            print(
                f"  {op_name:>8}: n={m['count']}  "
                f"avg_time={m['avg_time']:.2f}s  total_time={m['total_time']:.2f}s  "
                f"avg_tokens={m['avg_tokens']:.0f}  total_tokens={m['total_tokens']}"
            )
    print("-" * 60)
