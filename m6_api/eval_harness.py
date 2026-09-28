"""
M6 Evaluation Harness
Implements metrics: Recall@5, BERTScore, BLEU, IoU
Run: python -m m6_api.eval_harness
"""
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from shared.schemas import QueryResponse, ResultItem, StructuredQuery, TaskType
from m5_controller.dispatch_table import orchestrate_async
from shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single query."""
    query_id: str
    query_text: str
    task_type: str
    recall_at_5: float = 0.0
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bertscore_f1: float = 0.0
    bleu_score: float = 0.0
    iou: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class EvalResult:
    """Aggregate evaluation results."""
    timestamp: str
    total_queries: int
    successful_queries: int
    metrics: List[EvalMetrics]
    aggregate: Dict[str, float]


class MockEmbeddingModel:
    """Mock embedding model for semantic similarity (replace with real model)."""
    
    def encode(self, texts: List[str]) -> np.ndarray:
        # Simple hash-based embeddings for testing
        embeddings = []
        for text in texts:
            # Create deterministic pseudo-embedding from text hash
            hash_val = hash(text)
            np.random.seed(abs(hash_val) % (2**32))
            emb = np.random.randn(384)  # 384-dim like MiniLM
            embeddings.append(emb)
        return np.array(embeddings)


def compute_recall_at_5(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Compute Recall@5 for retrieval."""
    if not relevant_ids:
        return 1.0
    retrieved_set = set(retrieved_ids[:5])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def compute_bertscore(prediction: str, reference: str) -> Dict[str, float]:
    """Compute BERTScore (using mock embeddings for now)."""
    # In production, use: from bert_score import score
    # For now, use cosine similarity of mock embeddings
    model = MockEmbeddingModel()
    pred_emb = model.encode([prediction])
    ref_emb = model.encode([reference])
    
    cos_sim = cosine_similarity(pred_emb, ref_emb)[0][0]
    # Convert to precision/recall/f1-like scores
    precision = max(0.0, cos_sim)
    recall = max(0.0, cos_sim)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_bleu(prediction: str, reference: str) -> float:
    """Compute BLEU score (simplified n-gram overlap)."""
    # In production, use: from nltk.translate.bleu_score import sentence_bleu
    # Simplified version using word n-gram overlap
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    
    if not pred_words or not ref_words:
        return 0.0
    
    # 1-gram to 4-gram
    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = []
    
    for n in range(1, 5):
        if n > len(pred_words) or n > len(ref_words):
            precisions.append(0.0)
            continue
            
        pred_ngrams = set(tuple(pred_words[i:i+n]) for i in range(len(pred_words) - n + 1))
        ref_ngrams = set(tuple(ref_words[i:i+n]) for i in range(len(ref_words) - n + 1))
        
        if not pred_ngrams:
            precisions.append(0.0)
        else:
            overlap = len(pred_ngrams & ref_ngrams)
            precisions.append(overlap / len(pred_ngrams))
    
    # Geometric mean
    if all(p > 0 for p in precisions):
        bleu = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        bleu = 0.0
    
    return float(bleu)


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute Intersection over Union for two bounding boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


class EvalHarness:
    """Evaluation harness for SatQuery."""
    
    def __init__(self, eval_data_path: str = "fixtures/eval_queries.json"):
        self.eval_data_path = Path(eval_data_path)
        self.eval_queries = self._load_eval_queries()
    
    def _load_eval_queries(self) -> List[Dict[str, Any]]:
        """Load evaluation queries with ground truth."""
        if self.eval_data_path.exists():
            with open(self.eval_data_path) as f:
                return json.load(f)
        return []
    
    async def run_evaluation(self, max_queries: Optional[int] = None) -> EvalResult:
        """Run evaluation on all queries."""
        queries = self.eval_queries
        if max_queries:
            queries = queries[:max_queries]
        
        print(f"Running evaluation on {len(queries)} queries...")
        metrics = []
        
        for i, eval_item in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] {eval_item['query_text'][:60]}...")
            metric = await self._evaluate_single(eval_item)
            metrics.append(metric)
        
        # Compute aggregates
        successful = [m for m in metrics if m.success]
        aggregate = {}
        if successful:
            aggregate = {
                "recall_at_5": np.mean([m.recall_at_5 for m in successful]),
                "bertscore_precision": np.mean([m.bertscore_precision for m in successful]),
                "bertscore_recall": np.mean([m.bertscore_recall for m in successful]),
                "bertscore_f1": np.mean([m.bertscore_f1 for m in successful]),
                "bleu_score": np.mean([m.bleu_score for m in successful]),
                "iou": np.mean([m.iou for m in successful]),
                "avg_latency_ms": np.mean([m.latency_ms for m in successful]),
                "success_rate": len(successful) / len(metrics),
            }
        
        return EvalResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_queries=len(metrics),
            successful_queries=len(successful),
            metrics=metrics,
            aggregate=aggregate,
        )
    
    async def _evaluate_single(self, eval_item: Dict[str, Any]) -> EvalMetrics:
        """Evaluate a single query against ground truth."""
        query_text = eval_item["query_text"]
        task_type = eval_item.get("task_type", "search")
        ground_truth = eval_item.get("ground_truth", {})
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Run the query through the pipeline
            response = await orchestrate_async(query_text)
            
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Compute metrics based on task type
            if task_type == "search":
                return self._eval_search(response, ground_truth, query_text, task_type, latency_ms)
            elif task_type == "vqa":
                return self._eval_vqa(response, ground_truth, query_text, task_type, latency_ms)
            elif task_type == "caption":
                return self._eval_caption(response, ground_truth, query_text, task_type, latency_ms)
            elif task_type == "grounding":
                return self._eval_grounding(response, ground_truth, query_text, task_type, latency_ms)
            elif task_type == "change":
                return self._eval_change(response, ground_truth, query_text, task_type, latency_ms)
            elif task_type == "fusion":
                return self._eval_fusion(response, ground_truth, query_text, task_type, latency_ms)
            else:
                return EvalMetrics(
                    query_id=response.query_id,
                    query_text=query_text,
                    task_type=task_type,
                    latency_ms=latency_ms,
                    success=True,
                )
                
        except Exception as e:
            logger.error(f"Evaluation failed for query: {e}")
            return EvalMetrics(
                query_id=str(hash(query_text)),
                query_text=query_text,
                task_type=task_type,
                success=False,
                error=str(e),
            )
    
    def _eval_search(self, response: QueryResponse, gt: Dict, query_text: str, task_type: str, latency_ms: float) -> EvalMetrics:
        """Evaluate search task."""
        retrieved_ids = [r.image_id for r in response.results]
        relevant_ids = gt.get("relevant_scene_ids", [])
        
        recall = compute_recall_at_5(retrieved_ids, relevant_ids)
        
        return EvalMetrics(
            query_id=response.query_id,
            query_text=query_text,
            task_type=task_type,
            recall_at_5=recall,
            latency_ms=latency_ms,
            success=True,
        )
    
    def _eval_vqa(self, response: QueryResponse, gt: Dict, query_text: str, task_type: str, latency_ms: float) -> EvalMetrics:
        """Evaluate VQA task."""
        if not response.results:
            return EvalMetrics(query_id=response.query_id, query_text=query_text, task_type=task_type, latency_ms=latency_ms, success=False, error="No results")
        
        pred_answer = response.results[0].model_outputs.vqa_answer or ""
        ref_answer = gt.get("answer", "")
        
        bertscore = compute_bertscore(pred_answer, ref_answer)
        bleu = compute_bleu(pred_answer, ref_answer)
        
        return EvalMetrics(
            query_id=response.query_id,
            query_text=query_text,
            task_type=task_type,
            bertscore_precision=bertscore["precision"],
            bertscore_recall=bertscore["recall"],
            bertscore_f1=bertscore["f1"],
            bleu_score=bleu,
            latency_ms=latency_ms,
            success=True,
        )
    
    def _eval_caption(self, response: QueryResponse, gt: Dict, query_text: str, task_type: str, latency_ms: float) -> EvalMetrics:
        """Evaluate caption task."""
        if not response.results:
            return EvalMetrics(query_id=response.query_id, query_text=query_text, task_type=task_type, latency_ms=latency_ms, success=False, error="No results")
        
        pred_caption = response.results[0].model_outputs.caption or ""
        ref_caption = gt.get("caption", "")
        
        bertscore = compute_bertscore(pred_caption, ref_caption)
        bleu = compute_bleu(pred_caption, ref_caption)
        
        return EvalMetrics(
            query_id=response.query_id,
            query_text=query_text,
            task_type=task_type,
            bertscore_precision=bertscore["precision"],
            bertscore_recall=bertscore["recall"],
            bertscore_f1=bertscore["f1"],
            bleu_score=bleu,
            latency_ms=latency_ms,
            success=True,
        )
    
    def _eval_grounding(self, response: QueryResponse, gt: Dict, query_text: str, task_type: str, latency_ms: float) -> EvalMetrics:
        """Evaluate grounding task."""
        if not response.results:
            return EvalMetrics(query_id=response.query_id, query_text=query_text, task_type=task_type, latency_ms=latency_ms, success=False, error="No results")
        
        pred_bbox = response.results[0].model_outputs.grounding_bbox
        ref_bbox = gt.get("bbox", [0.25, 0.25, 0.75, 0.75])
        
        iou = 0.0
        if pred_bbox and isinstance(pred_bbox, dict):
            pred_coords = pred_bbox.get("bbox", pred_bbox)
            if isinstance(pred_coords, list) and len(pred_coords) == 4:
                iou = compute_iou(pred_coords, ref_bbox)
        
        return EvalMetrics(
            query_id=response.query_id,
            query_text=query_text,
            task_type=task_type,
            iou=iou,
            latency_ms=latency_ms,
            success=True,
        )
    
    def _eval_change(self, response: QueryResponse, gt: Dict, query_text: str, task_type: str, latency_ms: float) -> EvalMetrics:
        """Evaluate change detection task."""
        if not response.results:
            return EvalMetrics(query_id=response.query_id, query_text=query_text, task_type=task_type, latency_ms=latency_ms, success=False, error="No results")
        
        pred_desc = response.results[0].model_outputs.change_description or ""
        ref_desc = gt.get("change_description", "")
        
        bertscore = compute_bertscore(pred_desc, ref_desc)
        bleu = compute_bleu(pred_desc, ref_desc)
        
        return EvalMetrics(
            query_id=response.query_id,
            query_text=query_text,
            task_type=task_type,
            bertscore_precision=bertscore["precision"],
            bertscore_recall=bertscore["recall"],
            bertscore_f1=bertscore["f1"],
            bleu_score=bleu,
            latency_ms=latency_ms,
            success=True,
        )
    
    def _eval_fusion(self, response: QueryResponse, gt: Dict, query_text: str, task_type: str, latency_ms: float) -> EvalMetrics:
        """Evaluate fusion task."""
        if not response.results:
            return EvalMetrics(query_id=response.query_id, query_text=query_text, task_type=task_type, latency_ms=latency_ms, success=False, error="No results")
        
        pred_fusion = response.results[0].model_outputs.fusion_statement or ""
        ref_fusion = gt.get("fusion_statement", "")
        
        bertscore = compute_bertscore(pred_fusion, ref_fusion)
        bleu = compute_bleu(pred_fusion, ref_fusion)
        
        return EvalMetrics(
            query_id=response.query_id,
            query_text=query_text,
            task_type=task_type,
            bertscore_precision=bertscore["precision"],
            bertscore_recall=bertscore["recall"],
            bertscore_f1=bertscore["f1"],
            bleu_score=bleu,
            latency_ms=latency_ms,
            success=True,
        )
    
    def save_results(self, result: EvalResult, output_path: str = "eval_results.json"):
        """Save evaluation results to JSON."""
        output = {
            "timestamp": result.timestamp,
            "total_queries": result.total_queries,
            "successful_queries": result.successful_queries,
            "aggregate": result.aggregate,
            "per_query": [asdict(m) for m in result.metrics],
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {output_path}")


def create_sample_eval_data(output_path: str = "fixtures/eval_queries.json"):
    """Create sample evaluation data for testing."""
    eval_queries = [
        {
            "query_text": "Show me Sentinel-2 images of agriculture near Delhi from June 2024",
            "task_type": "search",
            "ground_truth": {
                "relevant_scene_ids": ["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
            },
        },
        {
            "query_text": "What is the land cover in this image?",
            "task_type": "vqa",
            "ground_truth": {
                "answer": "The land cover is agriculture with some urban areas.",
            },
        },
        {
            "query_text": "Describe the land cover around Guwahati",
            "task_type": "caption",
            "ground_truth": {
                "caption": "This image shows urban areas around Guwahati with the Brahmaputra river visible.",
            },
        },
        {
            "query_text": "Highlight the water body in the north east",
            "task_type": "grounding",
            "ground_truth": {
                "bbox": [0.3, 0.2, 0.7, 0.6],
            },
        },
        {
            "query_text": "What changed between August 2023 and February 2024 in the Assam region?",
            "task_type": "change",
            "ground_truth": {
                "change_description": "Significant flooding detected in the Assam region between the two time periods.",
            },
        },
        {
            "query_text": "Combine optical and SAR observations for the forest area",
            "task_type": "fusion",
            "ground_truth": {
                "fusion_statement": "Optical shows dense forest canopy; SAR reveals forest structure and moisture content.",
            },
        },
    ]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_queries, f, indent=2)
    
    print(f"Created sample eval data at {output_path}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="M6 Evaluation Harness")
    parser.add_argument("--eval-data", default="fixtures/eval_queries.json", help="Evaluation data path")
    parser.add_argument("--output", default="eval_results.json", help="Output results path")
    parser.add_argument("--max-queries", type=int, help="Max queries to evaluate")
    parser.add_argument("--create-sample", action="store_true", help="Create sample eval data")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_eval_data(args.eval_data)
        return
    
    harness = EvalHarness(args.eval_data)
    result = await harness.run_evaluation(args.max_queries)
    harness.save_results(result, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Queries: {result.total_queries}")
    print(f"Successful: {result.successful_queries}")
    print(f"Success Rate: {result.aggregate.get('success_rate', 0):.2%}")
    print()
    for key, value in result.aggregate.items():
        if key != "success_rate":
            print(f"  {key}: {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())