"""
Validate JSON schemas for CI.
Run: python -m schemas.validate_schemas
"""
import json
import sys
from pathlib import Path


def validate_schema_file(schema_path: Path) -> tuple[bool, str]:
    """Validate a single JSON schema file."""
    try:
        with open(schema_path) as f:
            schema = json.load(f)

        # Basic validation - check required fields
        if "type" not in schema and "enum" not in schema:
            return False, "Missing 'type' or 'enum' field"

        # Check for title and description
        if "title" not in schema:
            return False, "Missing 'title' field"

        return True, "OK"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def main():
    schema_dir = Path(__file__).parent
    schema_files = list(schema_dir.glob("*.schema.json"))

    if not schema_files:
        print("No schema files found!")
        return 1

    print(f"Validating {len(schema_files)} schema files...")
    all_passed = True

    for schema_file in sorted(schema_files):
        passed, message = validate_schema_file(schema_file)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {schema_file.name}: {message}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll schemas validated successfully!")
        return 0
    else:
        print("\nSome schemas failed validation!")
        return 1


if __name__ == "__main__":
    sys.exit(main())