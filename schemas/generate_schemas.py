"""
Generate JSON schemas from Pydantic models.
Run: python -m schemas.generate_schemas
"""
import json
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.schemas import (
    SceneMetadata,
    StructuredQuery,
    ResultItem,
    ExecutionTrace,
    UniformError,
    ChangeResult,
    QueryResponse,
    AOI,
    ScoreBreakdown,
    ModelOutputs,
    ToolCall,
    TaskType,
    Sensor,
    ErrorType,
)


def generate_all_schemas():
    """Generate JSON schema files for all core schemas."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    schemas = {
        "scene_metadata": SceneMetadata,
        "structured_query": StructuredQuery,
        "result_item": ResultItem,
        "execution_trace": ExecutionTrace,
        "uniform_error": UniformError,
        "change_result": ChangeResult,
        "query_response": QueryResponse,
        "aoi": AOI,
        "score_breakdown": ScoreBreakdown,
        "model_outputs": ModelOutputs,
        "tool_call": ToolCall,
    }

    for name, model in schemas.items():
        schema = model.model_json_schema()
        # Add title and description
        schema["title"] = name.replace("_", " ").title()
        schema["description"] = f"SatQuery {name.replace('_', ' ')} schema"

        output_file = output_dir / f"{name}.schema.json"
        with open(output_file, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"Generated {output_file}")

    # Generate enum schemas
    enum_schemas = {
        "task_type": TaskType,
        "sensor": Sensor,
        "error_type": ErrorType,
    }

    for name, enum_class in enum_schemas.items():
        schema = {
            "type": "string",
            "enum": [e.value for e in enum_class],
            "title": name.replace("_", " ").title(),
            "description": f"SatQuery {name} enumeration",
        }
        output_file = output_dir / f"{name}.schema.json"
        with open(output_file, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"Generated {output_file}")

    print("All schemas generated successfully!")


if __name__ == "__main__":
    generate_all_schemas()