#!/usr/bin/env python3
"""Evaluate predeclared P/C/D samples; does not generate or silently drop results."""
import argparse
import hashlib
import json
from pathlib import Path

from itm.data.manifest import read_jsonl
from itm.metrics.rework_protocol import evaluate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("docs/itm_rework/protocol/result.schema.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path; existing results are immutable")
    records = list(read_jsonl(args.records))
    schema = json.loads(args.schema.read_text())
    for record in records:
        missing = set(schema["required"]) - record.keys()
        if missing:
            raise ValueError(f"missing per-sample provenance fields: {sorted(missing)}")
        for field, rule in schema.get("properties", {}).items():
            value = record.get(field)
            if "enum" in rule and value not in rule["enum"]:
                raise ValueError(f"invalid schema enum: {field}")
            allowed = rule.get("type")
            if allowed:
                allowed = [allowed] if isinstance(allowed, str) else allowed
                kind = "null" if value is None else "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else "string" if isinstance(value, str) else "other"
                if kind not in allowed:
                    raise ValueError(f"invalid schema type: {field}")
    result = evaluate_records(list(read_jsonl(args.expected)), records,
                              json.loads(args.thresholds.read_text()))
    result["input_hashes"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (args.expected, args.records, args.thresholds, args.schema)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
