"""End-to-end CPU accounting fixture; these scores are not research evidence."""
import json
from pathlib import Path
import subprocess
import sys


def test_cpu_runner_executes_evaluation_and_preserves_metrics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "docs/itm_rework/protocol/result.schema.json").read_text())
    record = {key: "synthetic_fixture" for key in schema["required"]}
    record.update(case_id="fixture", model_id="fixture", checkpoint_hash="fixture", source_group_id="fixture",
        protocol="C", sensor_config="head", split_role="val", sample_seed=1, train_seed=2,
        compatibility_label="compatible", witness_evidence_hash="fixture", thresholds_version="fixture",
        semantic_evaluator_version="fixture", quality_evaluator_version="fixture",
        semantic_metrics={"score": .8, "expected_direction_delta": .2},
        observation_metrics={"joint_error_m": .01}, quality_metrics={"score": .8}, failure_reason=None)
    record["joint_success"] = None  # Computed by the evaluator, never trusted from input.
    expected = tmp_path / "expected.jsonl"; expected.write_text(json.dumps(record)+"\n")
    records = tmp_path / "records.jsonl"; records.write_text(json.dumps(record)+"\n")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps(dict(status="frozen", fit_split="val", version="fixture",
        validation_evidence="synthetic-fixture-not-research", semantic_evaluator_version="fixture",
        quality_evaluator_version="fixture", observation_space="joint_proxy", semantic_min=.5,
        edit_delta_min=.1, quality_min=.5, observation_max={"joint_error_m":.05})))
    job = dict(job_id="evaluate_fixture", device="cpu", script="evaluate_rework_records.py", timeout_seconds=30,
        output_kind="file", arguments=["--expected",str(expected),"--records",str(records),"--thresholds",str(thresholds)])
    queue = tmp_path / "queue.json"; queue.write_text(json.dumps({"jobs":[job]}))
    output = tmp_path / "run"
    command = [sys.executable, "scripts/run_rework_stage.py", "--queue",str(queue),"--output",str(output),"--execute"]
    result = subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=45)
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "evaluate_fixture.json").read_text())
    assert payload["joint_success_rate"] == 1
    assert payload["rows"][0]["observation_metrics"]["joint_error_m"] == .01
    assert len(payload["input_hashes"]) == 4
    # Reusing a run directory must fail, never overwrite evidence.
    again = subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=45)
    assert again.returncode != 0


def test_gpu_job_is_rejected_before_launch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"jobs":[dict(job_id="gpu",device="cuda:0",script="evaluate_rework_records.py",
        timeout_seconds=30,output_kind="file",arguments=[])]}))
    result = subprocess.run([sys.executable,"scripts/run_rework_stage.py","--queue",str(queue),"--output",str(tmp_path/"run"),"--execute"],
                            cwd=root,capture_output=True,text=True,timeout=45)
    assert result.returncode != 0
    assert not (tmp_path/"run").exists()
