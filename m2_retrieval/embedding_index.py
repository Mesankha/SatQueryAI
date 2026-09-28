"""
M2 Retrieval - Embedding Index (FAISS adapter wrapper).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QuestionItem:
    """Legacy academic question item."""
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingIndex:
    """Simple embedding index wrapper."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._items: Dict[str, QuestionItem] = {}
        self._vectors: List[List[float]] = []
        self._ids: List[str] = []

    def add(self, item: QuestionItem):
        """Add an item to the index."""
        if item.embedding is not None:
            self._items[item.id] = item
            self._ids.append(item.id)
            self._vectors.append(item.embedding)

    def search(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[QuestionItem, float]]:
        """Search for top-k similar items by cosine similarity."""
        import math
        norm_q = math.sqrt(sum(x*x for x in query_vec))
        if norm_q < 1e-9:
            return []
        results = []
        for item_id, vec in zip(self._ids, self._vectors):
            norm_v = math.sqrt(sum(x*x for x in vec))
            if norm_v < 1e-9:
                continue
            score = sum(a*b for a, b in zip(query_vec, vec)) / (norm_q * norm_v)
            results.append((self._items[item_id], score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def size(self) -> int:
        return len(self._ids)

    def save(self, path: str):
        """Save index to JSON."""
        import json
        data = {
            "dim": self.dim,
            "items": [
                {
                    "id": item.id,
                    "text": item.text,
                    "embedding": item.embedding,
                    "metadata": item.metadata,
                }
                for item in self._items.values()
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: str) -> bool:
        """Load index from JSON. Returns True on success."""
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            self.dim = data.get("dim", 512)
            self._items = {}
            self._vectors = []
            self._ids = []
            for item_data in data.get("items", []):
                item = QuestionItem(
                    id=item_data["id"],
                    text=item_data["text"],
                    embedding=item_data.get("embedding"),
                    metadata=item_data.get("metadata", {}),
                )
                self.add(item)
            return True
        except Exception:
            return False