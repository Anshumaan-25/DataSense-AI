#!/usr/bin/env python3
"""CLI runner for the DataSense AI TPC-H Benchmark Suite.

Usage:
    python3 scripts/run_benchmark.py
    python3 scripts/run_benchmark.py --output reports/benchmark_results.json
    python3 scripts/run_benchmark.py --output reports/run.json --provider gemini --retries 3
    python3 scripts/run_benchmark.py --benchmark data/my_custom_queries.json

Exit codes:
    0 — Benchmark completed (regardless of Pass@N score).
    1 — Fatal setup error (missing file, DB init failure, etc.).
    2 — Evaluation/provider errors, including an aborted provider-failed run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when invoked from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_connection, init_tpch_db
from src.retriever import SchemaMetricRetriever
from src.evaluator import BenchmarkRunner, BenchmarkReport
from src.config import settings


# ─── ANSI colour helpers (gracefully disabled on non-TTY environments) ───────

def _supports_color() -> bool:
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if _supports_color():
        return f"\033[{code}m{text}\033[0m"
    return text


BOLD   = lambda t: _c(t, "1")
GREEN  = lambda t: _c(t, "32")
RED    = lambda t: _c(t, "31")
YELLOW = lambda t: _c(t, "33")
CYAN   = lambda t: _c(t, "36")
DIM    = lambda t: _c(t, "2")


# ─── Progress callback (prints inline status to terminal) ────────────────────

def make_progress_callback(total: int):
    """Return a callback that prints a live progress line."""
    bar_width = 30

    def callback(current: int, total_tasks: int, task_id: str):
        filled = int(bar_width * current / total_tasks)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = current / total_tasks * 100
        line = f"\r  [{bar}] {pct:5.1f}%  ({current}/{total_tasks})  {DIM(task_id):<16}"
        sys.stdout.write(line)
        sys.stdout.flush()
        if current == total_tasks:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return callback


# ─── Pretty summary printer ──────────────────────────────────────────────────

def print_summary(report: BenchmarkReport, output_path: Path) -> None:
    """Print a formatted benchmark summary table to stdout."""
    divider = "─" * 60

    print(f"\n{BOLD('═' * 60)}")
    print(BOLD(CYAN("  📊  DataSense AI — TPC-H Benchmark Results")))
    print(f"{BOLD('═' * 60)}\n")

    # Aggregate metrics table
    rows = [
        ("Total Tasks Evaluated",        f"{report.total_tasks}"),
        ("EX-Eligible Tasks",            f"{report.ex_eligible_tasks}"),
        ("Pass@1 Accuracy",
            GREEN(f"{report.pass_at_1_accuracy:.1f}%") if report.pass_at_1_accuracy >= 70
            else YELLOW(f"{report.pass_at_1_accuracy:.1f}%")),
        ("Pass@N Final Accuracy",
            GREEN(f"{report.pass_at_n_accuracy:.1f}%") if report.pass_at_n_accuracy >= 70
            else YELLOW(f"{report.pass_at_n_accuracy:.1f}%")),
        ("Self-Correction Lift Δ EX",
            GREEN(f"+{report.self_correction_delta:.1f}%") if report.self_correction_delta > 0
            else DIM(f"{report.self_correction_delta:.1f}%")),
        ("Semantic Repairs",             f"{report.semantic_repair_count}"),
        ("Avg Query Latency",            f"{report.avg_latency_ms / 1000:.2f}s"),
        ("Ambiguity Classifier Accuracy",
            GREEN(f"{report.ambiguity_accuracy:.1f}%") if report.ambiguity_accuracy >= 70
            else YELLOW(f"{report.ambiguity_accuracy:.1f}%")),
    ]
    if report.aborted:
        print(RED(f"  Provider failure: stopped after {len(report.task_results)}/{report.total_tasks} tasks. Accuracy is unmeasured."))
        rows = [(label, "N/A" if label in ("Pass@1 Accuracy", "Pass@N Final Accuracy", "Self-Correction Lift Δ EX", "Ambiguity Classifier Accuracy") else value) for label, value in rows]
        rows[0] = ("Tasks Completed", f"{len(report.task_results)}/{report.total_tasks}")

    col_w = 34
    print(f"  {'Metric':<{col_w}}  {'Value'}")
    print(f"  {divider}")
    for label, value in rows:
        # Strip ANSI for alignment calculation
        raw_label = label
        print(f"  {raw_label:<{col_w}}  {value}")

    # Per-task table
    print(f"\n  {BOLD('Per-Task Results')}")
    print(f"  {divider}")
    header = f"  {'ID':<12} {'Category':<26} {'P@1':>4} {'P@N':>4} {'Atts':>4} {'Lat(s)':>7}  Question"
    print(header)
    print(f"  {divider}")

    for t in report.task_results:
        if not t.ex_eligible:
            p1_str = DIM("N/A")
            pn_str = DIM("N/A")
        else:
            p1_str = GREEN("✓") if t.pass_at_1 else RED("✗")
            pn_str = GREEN("✓") if t.pass_at_n else RED("✗")

        atts = str(t.total_attempts) if t.ex_eligible else DIM("-")
        lat  = f"{t.latency_ms / 1000:.2f}"
        q    = t.question[:35] + "…" if len(t.question) > 36 else t.question
        print(f"  {t.task_id:<12} {t.category:<26} {p1_str:>4} {pn_str:>4} {atts:>4} {lat:>7}s  {DIM(q)}")

    print(f"\n  {DIM(f'Full report saved → {output_path}')}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataSense AI — TPC-H Execution Accuracy Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        default=str(PROJECT_ROOT / "data" / "benchmark_queries.json"),
        help="Path to benchmark JSON dataset (default: data/benchmark_queries.json)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "benchmark_results.json"),
        help="Output path for the JSON report (default: reports/benchmark_results.json)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["gemini", "groq", "openai"],
        help="LLM provider override. Defaults to auto-detection from .env.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max self-repair attempts per query for Pass@N (default: 3)",
    )
    parser.add_argument(
        "--no-retriever",
        action="store_true",
        help="Disable ChromaDB RAG retriever (useful for isolated LLM testing).",
    )
    parser.add_argument("--read-only", action="store_true", help="Open an existing DuckDB file in read-only mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.retries < 1:
        print(RED("✗ --retries must be at least 1."), file=sys.stderr)
        return 1
    provider = args.provider or settings.get_effective_provider()
    if not {"gemini": settings.gemini_api_key, "groq": settings.groq_api_key,
            "openai": settings.openai_api_key}.get(provider):
        print(RED("✗ Configure an API key for the selected provider before live benchmarking."), file=sys.stderr)
        return 1

    # Validate benchmark file
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(RED(f"✗  Benchmark file not found: {benchmark_path}"), file=sys.stderr)
        return 1
    try:
        with benchmark_path.open(encoding="utf-8") as handle:
            tasks = json.load(handle)
        if not isinstance(tasks, list):
            raise ValueError("Benchmark dataset must be a JSON list.")
    except (OSError, ValueError) as exc:
        print(RED(f"✗ Invalid benchmark dataset: {exc}"), file=sys.stderr)
        return 1
    total = len(tasks)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(BOLD(f"\n  DataSense AI — TPC-H Benchmark Suite"))
    print(DIM(f"  Benchmark: {benchmark_path}"))
    print(DIM(f"  Provider:  {provider}"))
    print(DIM(f"  Retries:   {args.retries}"))
    print()

    # Initialise DuckDB
    print("  ⏳ Initializing DuckDB TPC-H database…", end=" ", flush=True)
    try:
        conn = get_connection(read_only=args.read_only)
        init_tpch_db(conn)
        print(GREEN("done"))
    except Exception as e:
        print(RED(f"FAILED\n  {e}"), file=sys.stderr)
        return 1

    # Initialise retriever
    retriever = None
    if not args.no_retriever:
        print("  ⏳ Indexing ChromaDB retriever…", end=" ", flush=True)
        try:
            metrics_file = PROJECT_ROOT / "data" / "metrics.yaml"
            from src.database import get_schema_info
            schema_dict = get_schema_info(conn, as_string=False)
            retriever = SchemaMetricRetriever()
            retriever.index_catalog(str(metrics_file), schema_dict)
            print(GREEN("done"))
        except Exception as e:
            print(RED(f"FAILED ({e})"), file=sys.stderr)
            return 1

    print(f"\n  🚀 Running {total} benchmark tasks…\n")
    wall_start = time.perf_counter()

    runner = BenchmarkRunner(
        db_conn=conn,
        retriever=retriever,
        max_retries=args.retries,
        provider=args.provider,
    )

    progress_cb = make_progress_callback(total)

    try:
        report = runner.run_benchmark(
            benchmark_file=str(benchmark_path),
            progress_callback=progress_cb,
        )
    except Exception as e:
        print(RED(f"\n  ✗  Benchmark run failed: {e}"), file=sys.stderr)
        return 1

    wall_elapsed = time.perf_counter() - wall_start
    print(DIM(f"  Wall-clock time: {wall_elapsed:.1f}s"))

    # Save JSON report
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
    except Exception as e:
        print(RED(f"  ✗  Failed to write report: {e}"), file=sys.stderr)
        return 1

    # Print summary table
    print_summary(report, output_path)
    if report.evaluation_error_count:
        print(RED(f"  {report.evaluation_error_count} task(s) had evaluation/provider errors; inspect the report."), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
