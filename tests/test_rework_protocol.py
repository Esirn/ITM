from copy import deepcopy

import numpy as np
import pytest

from itm.metrics.rework_protocol import evaluate_records, grouped_bootstrap, valid_pair_diversity


@pytest.fixture
def example():
    thresholds = dict(status="frozen", fit_split="val", version="synthetic-test-v1",
        validation_evidence="synthetic-unit-test-only", semantic_evaluator_version="synthetic-semantic-v1",
        quality_evaluator_version="synthetic-quality-v1", observation_space="joint_proxy",
        semantic_min=.5, edit_delta_min=.1, quality_min=.5, observation_max={"joint_error_m": .05})
    plan = [dict(case_id=str(i), model_id="test-model", checkpoint_hash="fixture-checkpoint",
        sensor_config="head", condition="paired", sample_seed=1, protocol="C", split_role="val",
        source_group_id=str(i), compatibility_label="compatible", witness_evidence_hash="synthetic-witness",
        random_plan_id="synthetic-rng-v1", protocol_version="synthetic-v1", thresholds_version="synthetic-test-v1") for i in range(3)]
    records = [{**row, "semantic_metrics": {"score": .8, "expected_direction_delta": .2},
        "observation_metrics": {"joint_error_m": .02}, "quality_metrics": {"score": .8},
        "semantic_evaluator_version": "synthetic-semantic-v1", "quality_evaluator_version": "synthetic-quality-v1"} for row in plan]
    return plan, records, thresholds


def test_failures_and_missing_outputs_stay_in_denominator(example):
    plan, records, thresholds = example
    records[1]["semantic_metrics"]["score"] = np.nan
    report = evaluate_records(plan, records[:2], thresholds)
    assert report["samples"] == 3
    assert report["joint_success_rate"] == 1/3
    assert report["failure_rate"] == 2/3
    assert report["rows"][2]["failure_reason"] == "missing_output"
    assert report["rows"][0]["observation_metrics"]["joint_error_m"] == .02
    assert report["rows"][1]["semantic_metrics"]["score"] is None


def test_positive_but_too_small_edit_does_not_succeed(example):
    plan, records, thresholds = example
    records[0]["semantic_metrics"]["expected_direction_delta"] = .001
    assert evaluate_records(plan, records, thresholds)["joint_success_rate"] == 2/3


@pytest.mark.parametrize("change", ["mixed_checkpoint", "conflict_in_C", "duplicate", "unfrozen", "IMU", "wrong_rng"])
def test_protocol_violations_are_rejected(example, change):
    plan, records, thresholds = deepcopy(example)
    if change == "mixed_checkpoint": plan[0]["checkpoint_hash"] = "another"
    elif change == "conflict_in_C": plan[0]["compatibility_label"] = "incompatible"
    elif change == "duplicate": records.append(records[0])
    elif change == "unfrozen": thresholds["status"] = "draft"
    elif change == "IMU": thresholds["observation_space"] = "IMU"
    elif change == "wrong_rng": records[0]["random_plan_id"] = "changed"
    with pytest.raises(ValueError): evaluate_records(plan, records, thresholds)


def test_D_reports_components_without_compatible_JSR(example):
    plan, records, thresholds = example
    for row in plan + records:
        row.update(protocol="D", compatibility_label="uncertain")
    report = evaluate_records(plan, records, thresholds)
    assert report["joint_success_rate"] is None
    assert report["metric_name"] is None
    assert all(row["joint_success"] is None for row in report["rows"])


def test_P_does_not_require_edit_magnitude(example):
    plan, records, thresholds = example
    for row in plan + records: row["protocol"] = "P"
    for row in records: del row["semantic_metrics"]["expected_direction_delta"]
    assert evaluate_records(plan, records, thresholds)["joint_success_rate"] == 1


def test_group_weighting_does_not_count_repeated_seeds_as_new_cases():
    rows = [{"case_id": "a", "source_group_id": "g1", "delta": 0.}]*10
    rows += [{"case_id": "b", "source_group_id": "g1", "delta": 2.},
             {"case_id": "c", "source_group_id": "g2", "delta": 3.}]
    result = grouped_bootstrap(rows, "delta", samples=100)
    assert result["mean"] == 2.  # g1: mean(0,2)=1; g2:3
    assert result["source_groups"] == 2 and result["cases"] == 3


def test_diversity_requires_two_valid_samples():
    report = valid_pair_diversity([[0], [1], [3]], [True, False, False])
    assert report["status"] == "not_estimable"
    assert report["valid_pair_distance"] is None
    assert report["valid_fraction"] == 1/3
    assert report["all_finite_pair_distance"] == 2
