#!/usr/bin/env python3
"""Main entry-point for the LangChain SQL Agent evaluation.

Usage
-----
    # Small-sample test on both datasets (default)
    python main.py

    # Full evaluation
    python main.py --full

    # Single dataset, custom sample size
    python main.py --dataset locomo --sample-size 50

    # Verbose (print agent intermediate steps)
    python main.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime

import config


def _save_report(report: dict, tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{tag}_{ts}.json"
    path = os.path.join(config.RESULTS_DIR, fname)

    def _default(o):
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_default)
    return path


async def _async_main(args) -> None:
    results = {}

    if args.dataset in ("locomo", "all"):
        from run_locomo import run_experiment as run_locomo

        sample = 9999 if args.full else args.sample_size
        report = await run_locomo(sample_size=sample, verbose=args.verbose)
        path = _save_report(report, "locomo")
        print(f"\n[saved] {path}")
        results["locomo"] = report

    if args.dataset in ("syllabusqa", "all"):
        from run_syllabusqa import run_experiment as run_syllabusqa

        sample = 9999 if args.full else args.sample_size
        report = await run_syllabusqa(sample_size=sample, verbose=args.verbose)
        path = _save_report(report, "syllabusqa")
        print(f"\n[saved] {path}")
        results["syllabusqa"] = report

    if args.dataset in ("financebench", "all"):
        from run_financebench import run_experiment as run_financebench

        sample = 9999 if args.full else args.sample_size
        report = await run_financebench(sample_size=sample, verbose=args.verbose)
        path = _save_report(report, "financebench")
        print(f"\n[saved] {path}")
        results["financebench"] = report

    if args.dataset in ("qasper", "all"):
        from run_qasper import run_experiment as run_qasper

        sample = 9999 if args.full else args.sample_size
        report = await run_qasper(sample_size=sample, verbose=args.verbose)
        path = _save_report(report, "qasper")
        print(f"\n[saved] {path}")
        results["qasper"] = report

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for ds, r in results.items():
        qm = r["qa_metrics"]
        ins = r.get("insert_metrics", {})
        ret = r.get("retrieve_metrics", {})
        dl = r.get("delete_metrics", {})
        print(f"\n  [{ds.upper()}]")
        print(f"    F1={qm['f1']:.4f}  Recall={qm['recall']:.4f}  Accuracy={qm['accuracy']:.4f}")
        if ret:
            print(
                f"    Retrieve: avg_time={ret['avg_time']:.2f}s  "
                f"avg_tokens={ret['avg_tokens']:.0f}  total_tokens={ret['total_tokens']}"
            )
        if ins:
            print(
                f"    Insert:   avg_time={ins['avg_time']:.2f}s  "
                f"avg_tokens={ins['avg_tokens']:.0f}  total_tokens={ins['total_tokens']}"
            )
        if dl:
            print(
                f"    Delete:   avg_time={dl['avg_time']:.2f}s  "
                f"avg_tokens={dl['avg_tokens']:.0f}  total_tokens={dl['total_tokens']}"
            )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain SQL Agent Evaluation")
    parser.add_argument(
        "--dataset",
        choices=["locomo", "syllabusqa", "financebench", "qasper", "all"],
        default="all",
        help="Which dataset(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=config.SMALL_SAMPLE_SIZE,
        help="Number of samples per operation (default: %d)" % config.SMALL_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation (overrides --sample-size with full dataset size)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print agent steps")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
