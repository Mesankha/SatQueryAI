"""
M2 Retrieval - Satellite Index (Stage 2: Semantic Ranking).

Contains:
  - StructuredQuery / SceneMetadata dataclasses
  - BaseSatelliteEncoder + RemoteCLIP/GeoRSCLIP encoders
  - FAISSIndexFlatIPAdapter (FAISS inner product index)
  - SemanticEncoderPipeline (encoder fallback chain)
  - SatelliteSemanticRanker (Stage 2)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import os


def normalize_l2(vector: List[float]) -> List[float]:
    """L2-normalize a vector to unit norm."""
    import math
    norm = math.sqrt(sum(x * x for x in vector))
    if norm < 1e-9:
        return vector
    return [x / norm for x in vector]


@dataclass
class StructuredQuery:
    """Canonical StructuredQuery contract from M1 to M2."""
    text_query: Optional[str] = None
    aoi: Optional[Any] = None
    date_range: Optional[Any] = None
    sensor: Optional[Union[str, List[str]]] = None
    object: Optional[Union[str, List[str]]] = None
    cloud_cover_max: Optional[float] = None
    resolution_max: Optional[float] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = dict(self.extra_params)
        res.update({
            "text_query": self.text_query,
            "aoi": self.aoi,
            "date_range": self.date_range,
            "sensor": self.sensor,
            "object": self.object,
            "cloud_cover_max": self.cloud_cover_max,
            "resolution_max": self.resolution_max,
        })
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredQuery":
        if isinstance(data, cls):
            return data
        known_keys = {
            "text_query", "query", "text", "aoi", "bbox", "geometry",
            "date_range", "date", "dates", "sensor", "sensor_type",
            "object", "target_object", "objects", "cloud_cover_max",
            "max_cloud_cover", "resolution_max", "max_resolution", "extra_params"
        }
        text_query = data.get("text_query", data.get("query", data.get("text")))
        aoi = data.get("aoi", data.get("bbox", data.get("geometry")))
        date_range = data.get("date_range", data.get("date", data.get("dates")))
        sensor = data.get("sensor", data.get("sensor_type"))
        obj = data.get("object", data.get("target_object", data.get("objects")))
        cloud_max = data.get("cloud_cover_max", data.get("max_cloud_cover"))
        if cloud_max is not None:
            try:
                cloud_max = float(cloud_max)
            except (ValueError, TypeError):
                cloud_max = None
        res_max = data.get("resolution_max", data.get("max_resolution"))
        if res_max is not None:
            try:
                res_max = float(res_max)
            except (ValueError, TypeError):
                res_max = None
        extra = {k: v for k, v in data.items() if k not in known_keys}
        if "extra_params" in data and isinstance(data["extra_params"], dict):
            extra.update(data["extra_params"])
        return cls(
            text_query=text_query,
            aoi=aoi,
            date_range=date_range,
            sensor=sensor,
            object=obj,
            cloud_cover_max=cloud_max,
            resolution_max=res_max,
            extra_params=extra,
        )


@dataclass
class SceneMetadata:
    """Catalog scene metadata for satellite Earth-observation images."""
    scene_id: str
    aoi: Optional[List[float]] = None
    acquisition_date: Optional[str] = None
    sensor: Optional[str] = None
    cloud_cover: Optional[float] = None
    resolution: Optional[float] = None
    target_objects: List[str] = field(default_factory=list)
    score: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.extra_attributes)
        data.update({
            "scene_id": self.scene_id,
            "aoi": self.aoi,
            "acquisition_date": self.acquisition_date,
            "sensor": self.sensor,
            "cloud_cover": self.cloud_cover,
            "resolution": self.resolution,
            "target_objects": self.target_objects,
        })
        if self.score is not None:
            data["score"] = self.score
            data["composite_score"] = self.score
        if self.score_breakdown is not None:
            data["score_breakdown"] = self.score_breakdown
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneMetadata":
        known_keys = {
            "scene_id", "id", "aoi", "bbox", "footprint", "acquisition_date",
            "date", "sensor", "sensor_type", "cloud_cover", "resolution",
            "target_objects", "objects", "score", "composite_score", "score_breakdown", "extra_attributes"
        }
        scene_id = str(data.get("scene_id", data.get("id", "")))
        aoi = data.get("aoi", data.get("bbox", data.get("footprint")))
        acquisition_date = data.get("acquisition_date", data.get("date"))
        sensor = data.get("sensor", data.get("sensor_type"))
        cloud_cover = data.get("cloud_cover")
        if cloud_cover is not None:
            try:
                cloud_cover = float(cloud_cover)
            except (ValueError, TypeError):
                cloud_cover = None
        resolution = data.get("resolution")
        if resolution is not None:
            try:
                resolution = float(resolution)
            except (ValueError, TypeError):
                resolution = None
        target_objects = list(data.get("target_objects", data.get("objects", [])))
        score = data.get("score", data.get("composite_score"))
        if score is not None:
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = None
        score_breakdown = data.get("score_breakdown")
        extra = {k: v for k, v in data.items() if k not in known_keys}
        if "extra_attributes" in data and isinstance(data["extra_attributes"], dict):
            extra.update(data["extra_attributes"])
        return cls(
            scene_id=scene_id,
            aoi=aoi,
            acquisition_date=acquisition_date,
            sensor=sensor,
            cloud_cover=cloud_cover,
            resolution=resolution,
            target_objects=target_objects,
            score=score,
            score_breakdown=score_breakdown,
            extra_attributes=extra,
        )


class BaseSatelliteEncoder:
    """Base class for satellite encoders."""

    def encode_text(self, text: str) -> Tuple[List[float], str]:
        """Encode text to vector. Returns (vector, encoder_name)."""
        raise NotImplementedError


class DeterministicSatelliteEncoderFallback(BaseSatelliteEncoder):
    """Deterministic fallback encoder that produces zero vectors."""

    def encode_text(self, text: str) -> Tuple[List[float], str]:
        # Deterministic hash-based pseudo-embedding for reproducible results
        h = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to 512-dim normalized vector
        vec = []
        for i in range(0, len(h), 2):
            vec.append(int(h[i:i+2], 16) / 255.0 - 0.5)
        # Pad to 512
        while len(vec) < 512:
            vec.append(0.0)
        vec = normalize_l2(vec[:512])
        return vec, "deterministic_fallback"


class RemoteCLIPEncoder(BaseSatelliteEncoder):
    """RemoteCLIP encoder - production version requires actual model.

    This is a stub that uses the deterministic fallback. The real encoder
    is loaded lazily when the model weights are available.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self._fallback = DeterministicSatelliteEncoderFallback()

    def _try_load_model(self):
        """Try to load actual RemoteCLIP model."""
        # This would load the actual model in production
        # For now, return None to use fallback
        return None

    def encode_text(self, text: str) -> Tuple[List[float], str]:
        if self.model is None:
            self.model = self._try_load_model()
        if self.model is not None:
            # Use actual model encoding
            try:
                import numpy as np
                inputs = self.model.get_text_features(text)
                vec = inputs[0].cpu().numpy().tolist()
                return normalize_l2(vec), "remoteclip"
            except Exception:
                pass
        return self._fallback.encode_text(text)


class GeoRSCLIPEncoder(BaseSatelliteEncoder):
    """GeoRSCLIP encoder - fallback for RemoteCLIP."""

    def __init__(self):
        self._fallback = DeterministicSatelliteEncoderFallback()

    def encode_text(self, text: str) -> Tuple[List[float], str]:
        return self._fallback.encode_text(text)


class FAISSIndexFlatIPAdapter:
    """FAISS IndexFlatIP adapter with transparent in-memory fallback."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._index = None
        self._vectors: List[List[float]] = []
        self._ids: List[str] = []
        self._init_faiss()

    def _init_faiss(self):
        """Try to initialize FAISS, fall back to in-memory."""
        try:
            import faiss
            import numpy as np
            self._index = faiss.IndexFlatIP(self.dim)
            self._use_faiss = True
        except ImportError:
            self._use_faiss = False

    def add(self, vec_id: str, vec: List[float]):
        """Add a vector to the index."""
        norm_vec = normalize_l2(vec)
        self._ids.append(vec_id)
        self._vectors.append(norm_vec)
        if self._use_faiss and self._index is not None:
            try:
                import numpy as np
                arr = np.array([norm_vec], dtype=np.float32)
                self._index.add(arr)
            except Exception:
                pass

    def search(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for top-k similar vectors. Returns list of (id, score)."""
        if not self._ids:
            return []
        norm_query = normalize_l2(query_vec)
        if self._use_faiss and self._index is not None and self._index.ntotal > 0:
            try:
                import numpy as np
                q = np.array([norm_query], dtype=np.float32)
                scores, indices = self._index.search(q, min(top_k, self._index.ntotal))
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self._ids):
                        results.append((self._ids[idx], float(score)))
                return results
            except Exception:
                pass
        # In-memory fallback: cosine similarity
        results = []
        for vec_id, vec in zip(self._ids, self._vectors):
            score = sum(a * b for a, b in zip(norm_query, vec))
            results.append((vec_id, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def size(self) -> int:
        return len(self._ids)


class SemanticEncoderPipeline:
    """Encoder pipeline with RemoteCLIP -> GeoRSCLIP -> fallback chain."""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.remoteclip = RemoteCLIPEncoder()
        self.georsclip = GeoRSCLIPEncoder()
        self.fallback = DeterministicSatelliteEncoderFallback()

    def encode_text(self, text: str) -> Tuple[List[float], str]:
        """Try encoders in order. Returns (vector, encoder_used)."""
        # Try RemoteCLIP first
        try:
            vec, name = self.remoteclip.encode_text(text)
            if name != "deterministic_fallback" and any(x != 0.0 for x in vec):
                return vec, name
        except Exception:
            pass
        # Try GeoRSCLIP
        try:
            vec, name = self.georsclip.encode_text(text)
            if any(x != 0.0 for x in vec):
                return vec, name
        except Exception:
            pass
        # Fallback
        return self.fallback.encode_text(text)


class SatelliteSemanticRanker:
    """Stage 2: Semantic ranking of Stage 1 surviving candidates."""

    def __init__(self, embedding_dim: int = 512, encoder_pipeline: Optional[SemanticEncoderPipeline] = None):
        self.embedding_dim = embedding_dim
        self.encoder_pipeline = encoder_pipeline or SemanticEncoderPipeline(embedding_dim=embedding_dim)

    @staticmethod
    def extract_semantic_query_text(query: Any) -> Optional[str]:
        """Extract combined semantic text from query."""
        if query is None:
            return None
        q_dict = query.to_dict() if hasattr(query, "to_dict") else (query if isinstance(query, dict) else {})
        parts = []
        text_q = q_dict.get("text_query", q_dict.get("query", q_dict.get("text")))
        if text_q and str(text_q).strip():
            parts.append(str(text_q).strip())
        obj_q = q_dict.get("object", q_dict.get("target_object"))
        if obj_q:
            if isinstance(obj_q, (list, tuple)):
                parts.append(" ".join(str(o) for o in obj_q if o))
            elif str(obj_q).strip():
                parts.append(str(obj_q).strip())
        if not parts:
            return None
        return " ".join(parts)

    def rank_candidates(
        self,
        candidates: List[Any],
        query: Any,
        top_k: Optional[int] = None
    ) -> Tuple[List[Any], str]:
        """Perform Stage 2 semantic ranking on Stage 1 candidate set."""
        # Rule 1: Empty candidate set
        if not candidates:
            return [], "no_candidates"

        # Rule 2: Extract semantic query text
        semantic_text = self.extract_semantic_query_text(query)
        if not semantic_text:
            for c in candidates:
                if hasattr(c, "score_breakdown"):
                    if c.score_breakdown is None:
                        c.score_breakdown = {}
                    c.score_breakdown["semantic_score"] = 0.0
                if hasattr(c, "score") and getattr(c, "score", None) is None:
                    c.score = 0.0
            return candidates, "skipped_no_semantic_content"

        # Encode query text
        query_vec, encoder_used = self.encoder_pipeline.encode_text(semantic_text)

        # If both model encoders failed
        if encoder_used == "metadata_only_fallback" or not any(x != 0.0 for x in query_vec):
            for c in candidates:
                if hasattr(c, "score_breakdown"):
                    if c.score_breakdown is None:
                        c.score_breakdown = {}
                    c.score_breakdown["semantic_score"] = 0.0
                if hasattr(c, "score") and getattr(c, "score", None) is None:
                    c.score = 0.0
            candidates.sort(
                key=lambda x: (
                    -(float(x.score) if hasattr(x, "score") and x.score is not None else 0.0),
                    str(getattr(x, "scene_id", ""))
                )
            )
            return candidates, "metadata_only_fallback"

        # Score candidates by cosine similarity to query
        scored = []
        for c in candidates:
            try:
                # Build candidate text from target_objects + scene_id
                cand_text = " ".join(getattr(c, "target_objects", []) or []) + " " + getattr(c, "scene_id", "")
                cand_vec, _ = self.encoder_pipeline.encode_text(cand_text)
                score = sum(a * b for a, b in zip(query_vec, cand_vec))
            except Exception:
                score = 0.0
            if hasattr(c, "score_breakdown"):
                if c.score_breakdown is None:
                    c.score_breakdown = {}
                c.score_breakdown["semantic_score"] = float(score)
            scored.append((c, float(score)))

        # Sort descending by score
        scored.sort(key=lambda x: (-x[1], str(getattr(x[0], "scene_id", ""))))

        if top_k is not None and top_k > 0 and top_k < len(scored):
            scored = scored[:top_k]

        return [c for c, _ in scored], encoder_used