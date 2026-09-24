"""P/C/D result accounting and source-group statistics, independent of models.

This module aggregates measured scores, it is not a semantic or naturalness
oracle. Evaluator provenance and validation-derived frozen thresholds are required.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


VERSION = "itm-protocol-metrics-v1"
KEY = ("case_id", "model_id", "sensor_config", "condition", "sample_seed")


def grouped_bootstrap(rows, metric, *, samples=2000, seed=20260924):
    """Average repeats per case, then equally weight source groups.

    Call with paired per-output differences to preserve model pairing. The
    estimand is explicitly group-weighted (not the historical motion-weighted CI).
    """
    cases, memberships = defaultdict(list), {}
    for row in rows:
        case, group = row["case_id"], row["source_group_id"]
        if case in memberships and memberships[case] != group:
            raise ValueError("case has inconsistent source group")
        memberships[case] = group
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError("nonfinite bootstrap value; declare failures before inference")
        cases[case].append(value)
    groups = defaultdict(list)
    for case, values in cases.items():
        groups[memberships[case]].append(float(np.mean(values)))
    values = np.array([np.mean(groups[key]) for key in sorted(groups)])
    if not len(values):
        return {"status": "not_estimable", "source_groups": 0, "ci95": None}
    result = {"status": "ok" if len(values) >= 2 else "not_estimable", "source_groups": len(values),
              "cases": len(cases), "mean": float(values.mean()), "ci95": None,
              "estimand": "equal_source_group_mean_of_case_means", "seed": seed,
              "bootstrap_samples": samples, "training_variance_included": False}
    if len(values) >= 2:
        if samples < 1:
            raise ValueError("bootstrap sample budget must be positive")
        rng = np.random.default_rng(seed)
        estimates = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(samples)]
        result["ci95"] = np.quantile(estimates, [.025, .975]).tolist()
    return result


def _identity(row):
    return tuple(row[key] for key in KEY)


def _index(rows):
    result = {}
    for row in rows:
        key = _identity(row)
        if key in result:
            raise ValueError(f"duplicate sample identity {key}")
        result[key] = row
    return result


def _finite_json(value):
    """Retain raw metrics without writing nonstandard NaN/Infinity JSON tokens."""
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def validate_thresholds(thresholds):
    if thresholds.get("status") != "frozen" or thresholds.get("fit_split") != "val":
        raise ValueError("thresholds must be frozen using validation data")
    for field in ("version", "validation_evidence", "semantic_evaluator_version", "quality_evaluator_version"):
        if not thresholds.get(field):
            raise ValueError(f"missing threshold provenance: {field}")
    if thresholds.get("observation_space") != "joint_proxy":
        raise ValueError("only joint_proxy is currently certified for protocol accounting; IMU certification pending")
    if not thresholds.get("observation_max"):
        raise ValueError("at least one explicit observation threshold required")
    numeric = [thresholds[k] for k in ("semantic_min", "edit_delta_min", "quality_min")]
    numeric.extend(thresholds["observation_max"].values())
    if not np.isfinite(np.asarray(numeric, dtype=float)).all() or thresholds["edit_delta_min"] <= 0:
        raise ValueError("finite thresholds and positive semantic edit magnitude required")
    if any(value < 0 for value in thresholds["observation_max"].values()):
        raise ValueError("observation error thresholds must be nonnegative")


def evaluate_records(expected, records, thresholds):
    """Use a predeclared denominator; missing/NaN/failed samples remain failures.

    One call handles one model/checkpoint, condition, configuration and protocol.
    Conflicts/uncertain cases belong in D; they never enter C's denominator.
    """
    validate_thresholds(thresholds)
    plan, actual = _index(expected), _index(records)
    if not plan:
        raise ValueError("empty evaluation plan")
    for row in expected:
        for field in ("source_group_id", "checkpoint_hash", "random_plan_id", "protocol_version", "thresholds_version"):
            if not row.get(field):
                raise ValueError(f"missing plan provenance: {field}")
        if row["thresholds_version"] != thresholds["version"]:
            raise ValueError("plan thresholds version mismatch")
        if row.get("split_role") not in {"val", "sealed_test", "historical_development_diagnostic"}:
            raise ValueError("explicit valid split role required")
    if actual.keys() - plan.keys():
        raise ValueError("unexpected result identities")
    scope_keys = ("model_id", "checkpoint_hash", "sensor_config", "condition", "protocol", "split_role")
    if len({tuple(row[k] for k in scope_keys) for row in expected}) != 1:
        raise ValueError("mixed model/checkpoint/configuration/protocol scope")
    first = expected[0]
    if first["sensor_config"] not in {"head", "wrists"}:
        raise ValueError("configuration outside the contract")
    protocol = first["protocol"]
    if protocol not in {"P", "C", "D"}:
        raise ValueError("unknown protocol")
    evaluated = []
    for key, case in plan.items():
        label = case.get("compatibility_label")
        if protocol == "C" and (label != "compatible" or not case.get("witness_evidence_hash")):
            raise ValueError("C requires compatible labels with frozen independent witness evidence")
        if protocol == "D" and label not in {"incompatible", "uncertain"}:
            raise ValueError("D contains only incompatible or uncertain cases")
        row = actual.get(key)
        reason = "missing_output" if row is None else row.get("failure_reason")
        successes = {"semantic_success": False, "observation_success": False, "quality_success": False}
        if row is not None:
            for field in (*scope_keys, "source_group_id", "random_plan_id", "protocol_version", "thresholds_version"):
                if row.get(field) != case.get(field):
                    raise ValueError(f"result provenance differs from plan: {field}")
            if row.get("thresholds_version") != thresholds["version"]:
                raise ValueError("wrong thresholds version")
            for field in ("semantic_evaluator_version", "quality_evaluator_version"):
                if row.get(field) != thresholds[field]:
                    raise ValueError(f"wrong {field}")
            if not reason:
                try:
                    semantic = float(row["semantic_metrics"]["score"])
                    delta = float(row["semantic_metrics"]["expected_direction_delta"]) if protocol == "C" else 0.
                    quality = float(row["quality_metrics"]["score"])
                    errors = {k: float(row["observation_metrics"][k]) for k in thresholds["observation_max"]}
                    finite = np.isfinite([semantic, delta, quality, *errors.values()]).all()
                    if not finite or any(x < 0 for x in errors.values()):
                        reason = "invalid_metrics"
                    else:
                        successes = {
                            "semantic_success": semantic >= thresholds["semantic_min"] and (protocol != "C" or delta >= thresholds["edit_delta_min"]),
                            "observation_success": all(v <= thresholds["observation_max"][k] for k, v in errors.items()),
                            "quality_success": quality >= thresholds["quality_min"],
                        }
                except (KeyError, TypeError, ValueError):
                    reason = "missing_or_invalid_metrics"
        joint_success = None if protocol == "D" else all(successes.values()) and not bool(reason)
        evaluated.append({**_finite_json(row or {}), **case, **successes, "joint_success": joint_success,
                          "failure_reason": reason})
    success = float(np.mean([row["joint_success"] for row in evaluated])) if protocol != "D" else None
    return {
        "evaluator_version": VERSION, "protocol": protocol, "samples": len(evaluated),
        "failure_rate": sum(bool(row["failure_reason"]) for row in evaluated) / len(evaluated),
        "metric_name": "JSR_joint_proxy" if protocol != "D" else None,
        "joint_success_rate": success if protocol != "D" else None,
        "component_success_rates": {key: float(np.mean([row[key] for row in evaluated])) for key in successes},
        "source_group_uncertainty": grouped_bootstrap(evaluated, "joint_success") if protocol != "D" else None,
        "rows": evaluated,
    }


def valid_pair_diversity(features, valid):
    """Explicit feature-space distance; never named null-space diversity."""
    array, mask = np.asarray(features, dtype=float), np.asarray(valid, dtype=bool)
    if array.ndim != 2 or mask.shape != (len(array),) or not len(array):
        raise ValueError("nonempty (samples, features) and matching validity required")
    if not np.isfinite(array[mask]).all():
        raise ValueError("valid output features must be finite")
    finite = np.isfinite(array).all(1)
    def pairs(a):
        values = [np.linalg.norm(a[i]-a[j]) for i in range(len(a)) for j in range(i)]
        return float(np.mean(values)) if values else None
    selected = array[mask]
    return {"valid_fraction": float(mask.mean()), "valid_samples": len(selected),
            "status": "ok" if len(selected) >= 2 else "not_estimable",
            "valid_pair_distance": pairs(selected), "all_finite_pair_distance": pairs(array[finite]),
            "nonfinite_fraction": float((~finite).mean())}
