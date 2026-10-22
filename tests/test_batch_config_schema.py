import json
from pathlib import Path

import jsonschema
import pytest

from ramappy.pipeline.configuration import batch_config_json_schema

SCHEMA_FILE = Path(__file__).parents[1] / "docs" / "schemas" / "batch_config.schema.json"


def _core_schema():
    # Round-trip through JSON to match the committed file.
    return json.loads(json.dumps(batch_config_json_schema(include_plugins=False), sort_keys=True))


def test_schema_is_valid_and_validates_configs():
    schema = _core_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    ok = {"input": {"format": "csv"}, "pipeline": [{"name": "normalize", "params": {}}]}
    assert list(validator.iter_errors(ok)) == []

    bad = {"input": {"format": "nope"}, "pipeline": [{"name": "nope"}]}
    assert list(validator.iter_errors(bad))


def test_committed_schema_is_up_to_date():
    committed = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    if committed != _core_schema():
        pytest.fail(
            "docs/schemas/batch_config.schema.json is stale. "
            "Regenerate with: python -m ramappy.pipeline.configuration docs/schemas/batch_config.schema.json"
        )
