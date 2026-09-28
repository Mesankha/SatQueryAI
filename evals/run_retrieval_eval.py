#!/usr/bin/env python
"""
run_retrieval_eval.py - Retrieval evaluation harness.

Reads catalog.db + eval_splits.json; for each test-AOI query (generated from the
held-out AOI's object/location labels), computes Recall@5, Precision@5, mAP@5,
cosine similarity stats, and p50/p95 latency of engine.retrieve; writes
evals/report_<YYYYMMDD>.json.

Refuses to run (exit code 2, clear message) if the index or splits file is missing.
Never fabricates a report.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from m2_retrieval.engine import retrieve
from shared.schemas import StructuredQuery, TaskType, Sensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_history(history_path: str) -> list[dict[str, Any]]:
    """Load evaluation history from JSONL file."""
    path = Path(history_path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_latest_baseline(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Get the latest baseline entry from history."""
    baselines = [entry for entry in history if entry.get("tag", "").startswith("baseline_")]
    if not baselines:
        return None
    # Return the most recent baseline (last in file)
    return baselines[-1]


def compare_with_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Compare current metrics with baseline."""
    comparison = {}
    for key in ["metrics", "cosine_similarity", "latency_ms"]:
        if key in current and key in baseline:
            comparison[key] = {}
            for metric, value in current[key].items():
                if metric in baseline[key]:
                    baseline_value = baseline[key][metric]
                    if baseline_value != 0:
                        delta_pct = ((value - baseline_value) / baseline_value) * 100
                    else:
                        delta_pct = 0.0
                    comparison[key][metric] = {
                        "current": value,
                        "baseline": baseline_value,
                        "delta_pct": delta_pct,
                    }
    return comparison


def load_eval_splits(splits_path: str) -> dict[str, Any]:
    """Load evaluation splits from JSON file."""
    if not Path(splits_path).exists():
        raise FileNotFoundError(f"Eval splits file not found: {splits_path}")
    with open(splits_path) as f:
        return json.load(f)
    import sqlite3
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Catalog database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scenes")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def generate_test_queries(splits: dict[str, Any], catalog: list[dict[str, Any]]) -> list[StructuredQuery]:
    """
    Generate test queries from held-out AOI scenes.
    For each test-AOI scene, create a query using its object/location.
    """
    # Group test scenes by AOI
    test_scenes = [s for s in catalog if splits.get(s["scene_id"], {}).get("split") == "test"]
    if not test_scenes:
        logger.warning("No test scenes found in eval splits")
        return []

    queries = []
    for scene in test_scenes:
        # Build query from scene metadata
        query_text_parts = []
        if scene.get("object"):
            query_text_parts.append(f"Show me {scene['object']}")
        else:
            query_text_parts.append("Show me images")

        if scene.get("sensor") == "Sentinel-2":
            query_text_parts.append("Sentinel-2")
        elif scene.get("sensor") == "Sentinel-1":
            query_text_parts.append("Sentinel-1")

        query_text = " ".join(query_text_parts)

        query = StructuredQuery(
            query_text=query_text,
            task_type=TaskType.search,
            sensor=Sensor.sentinel_2 if scene.get("sensor") == "Sentinel-2" else Sensor.sentinel_1,
            object=scene.get("object"),
            start_date=datetime.fromisoformat(scene["acquisition_time"].replace("Z", "+00:00")) if scene.get("acquisition_time") else None,
            end_date=None,
        )
        queries.append(query)

    return queries


def compute_metrics(
    retrieved: list[StructuredQuery],
    relevant: list[str],
    top_k: int = 5,
) -> dict[str, float]:
    """
    Compute Recall@k, Precision@k, mAP@k for a single query.
    """
    if not relevant:
        return {"recall@5": 0.0, "precision@5": 0.0, "map@5": 0.0}

    retrieved_ids = [r.scene_id for r in retrieved[:top_k]]
    relevant_set = set(relevant)

    # Recall@k
    hits = sum(1 for rid in retrieved_ids if rid in relevant_set)
    recall = hits / len(relevant_set) if relevant_set else 0.0

    # Precision@k
    precision = hits / len(retrieved_ids) if retrieved_ids else 0.0

    # mAP@k (average precision)
    ap = 0.0
    num_hits = 0
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            num_hits += 1
            ap += num_hits / (i + 1)
    map_score = ap / len(relevant_set) if relevant_set else 0.0

    return {
        "recall@5": recall,
        "precision@5": precision,
        "map@5": map_score,
    }


def run_evaluation(
    db_path: str,
    splits_path: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Run retrieval evaluation.

    Returns:
        Dict with metrics, cosine similarity stats, and latency stats.
    """
    splits = load_eval_splits(splits_path)
    catalog = load_catalog(db_path)

    # Build ground truth: relevant scene_ids per test AOI
    # For simplicity, relevant = all scenes from same AOI and sensor
    test_scenes = [s for s in catalog if splits.get(s["scene_id"], {}).get("split") == "test"]
    if not test_scenes:
        logger.warning("No test scenes found")
        return {"error": "No test scenes in eval splits"}

    # Group by AOI and sensor
    from collections import defaultdict
    gt_by_aoi_sensor = defaultdict(list)
    for scene in test_scenes:
        aoi = splits[scene["scene_id"]]["aoi"]
        sensor = scene["sensor"]
        gt_by_aoi_sensor[(aoi, sensor)].append(scene["scene_id"])

    queries = generate_test_queries(splits, catalog)
    if not queries:
        return {"error": "No queries generated"}

    all_metrics = []
    all_latencies = []
    all_cosine_sims = []

    for query in queries:
        # Determine ground truth for this query
        aoi = None
        for scene in catalog:
            if scene.get("object") == query.object and scene.get("sensor") == query.sensor.value:
                aoi = splits.get(scene["scene_id"], {}).get("aoi")
                break

        relevant = gt_by_aoi_sensor.get((aoi, query.sensor.value), []) if aoi else []

        # Run retrieval with timing
        start = time.perf_counter()
        retrieved = retrieve(query, db_path, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        all_latencies.append(latency_ms)

        # Compute metrics
        metrics = compute_metrics(retrieved, relevant, top_k)
        all_metrics.append(metrics)

        # Cosine similarity stats from retrieved scores
        scores = [getattr(r, "score", 0.5) for r in retrieved]
        if scores:
            all_cosine_sims.extend(scores)

    # Aggregate metrics
    avg_recall = np.mean([m["recall@5"] for m in all_metrics]) if all_metrics else 0.0
    avg_precision = np.mean([m["precision@5"] for m in all_metrics]) if all_metrics else 0.0
    avg_map = np.mean([m["map@5"] for m in all_metrics]) if all_metrics else 0.0

    # Latency stats
    p50_latency = np.percentile(all_latencies, 50) if all_latencies else 0.0
    p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0.0

    # Cosine similarity stats
    cosine_mean = np.mean(all_cosine_sims) if all_cosine_sims else 0.0
    cosine_std = np.std(all_cosine_sims) if all_cosine_sims else 0.0

    return {
        "metrics": {
            "recall@5": float(avg_recall),
            "precision@5": float(avg_precision),
            "map@5": float(avg_map),
        },
        "cosine_similarity": {
            "mean": float(cosine_mean),
            "std": float(cosine_std),
            "min": float(np.min(all_cosine_sims)) if all_cosine_sims else 0.0,
            "max": float(np.max(all_cosine_sims)) if all_cosine_sims else 0.0,
        },
        "latency_ms": {
            "p50": float(p50_latency),
            "p95": float(p95_latency),
            "mean": float(np.mean(all_latencies)) if all_latencies else 0.0,
        },
        "num_queries": len(queries),
        "num_test_scenes": len(test_scenes),
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation harness")
    parser.add_argument("--db", default="./data/catalog.db", help="Catalog database path")
    parser.add_argument("--splits", default="./data/eval_splits.json", help="Eval splits JSON path")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for metrics")
    parser.add_argument("--tag", default=None, help="Tag for this run (appended to history). First real run should use 'baseline_2026-09'")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare results against latest baseline in history.jsonl")
    parser.add_argument("--output", default=None, help="Output report path (default: evals/report_<date>.json)")

    args = parser.parse_args()

    # Check required files exist
    if not Path(args.db).exists():
        logger.error(f"Catalog database not found: {args.db}")
        sys.exit(2)

    if not Path(args.splits).exists():
        logger.error(f"Eval splits file not found: {args.splits}")
        sys.exit(2)

    logger.info(f"Running retrieval evaluation: db={args.db}, splits={args.splits}")

    try:
        results = run_evaluation(args.db, args.splits, args.top_k)

        if "error" in results:
            logger.error(results["error"])
            sys.exit(2)

        # Add metadata
        results["date"] = datetime.utcnow().isoformat() + "Z"
        results["tag"] = args.tag or f"eval_{datetime.utcnow().strftime('%Y%m%d')}"

        # Baseline comparison if requested
        if args.compare_baseline:
            history = load_history("./evals/history.jsonl")
            baseline = get_latest_baseline(history)
            if baseline:
                comparison = compare_with_baseline(results, baseline)
                results["baseline_comparison"] = comparison
                logger.info(f"Compared with baseline: {baseline.get('tag', 'unknown')}")
            else:
                logger.warning("No baseline found in history for comparison")

        # Write report
        output_path = args.output or f"./evals/report_{datetime.utcnow().strftime('%Y%m%d')}.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation complete. Report written to {output_path}")

        # Append to history if tag provided
        if args.tag:
            history_path = "./evals/history.jsonl"
            Path(history_path).parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "a") as f:
                f.write(json.dumps(results) + "\n")
            logger.info(f"Appended to history: {history_path}")

        print(json.dumps(results, indent=2))

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()