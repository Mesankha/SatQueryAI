"""Schema validation helpers."""
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import SceneMetadata, StructuredQuery, ResultItem, ExecutionTrace


def load_fixture_scenes(path: str | Path) -> list[SceneMetadata]:
    data = json.loads(Path(path).read_text())
    return [SceneMetadata.model_validate(d) for d in data]


def load_fixture_queries(path: str | Path) -> list[StructuredQuery]:
    data = json.loads(Path(path).read_text())
    return [StructuredQuery.model_validate(d) for d in data]


def load_fixture_results(path: str | Path) -> list[ResultItem]:
    data = json.loads(Path(path).read_text())
    return [ResultItem.model_validate(d) for d in data]


def validate_all_fixtures(fixtures_dir: str | Path) -> dict[str, Any]:
    fixtures_dir = Path(fixtures_dir)
    errors = []
    try:
        scenes = load_fixture_scenes(fixtures_dir / "fixtures_scenes.json")
    except ValidationError as e:
        errors.append(f"fixtures_scenes.json: {e}")
        scenes = []
    try:
        queries = load_fixture_queries(fixtures_dir / "fixtures_queries.json")
    except ValidationError as e:
        errors.append(f"fixtures_queries.json: {e}")
        queries = []
    try:
        results = load_fixture_results(fixtures_dir / "fixtures_results.json")
    except ValidationError as e:
        errors.append(f"fixtures_results.json: {e}")
        results = []

    if errors:
        raise ValueError("\n".join(errors))

    return {"scenes": scenes, "queries": queries, "results": results}
