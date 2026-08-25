#!/usr/bin/env python3
"""Score pre-generated HumanML 263D motions with the official evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from itm.data.manifest import read_jsonl
from evaluate_mdm_imu_control_generation import (
    EvalMotionDataset,
    _install_optional_dependency_stubs,
    _legacy_compatibility,
    aggregate_stats,
    collate_eval_batch,
    compute_metrics,
    first_caption,
    motion_item,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mdm-root", type=Path, required=True)
    parser.add_argument("--source-mean", type=Path, required=True)
    parser.add_argument("--source-std", type=Path, required=True)
    parser.add_argument("--t2m-mean", type=Path, required=True)
    parser.add_argument("--t2m-std", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diversity-times", type=int, default=30)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _legacy_compatibility()
    _install_optional_dependency_stubs()
    index = json.loads(args.generation_index.read_text())
    manifest = {str(item["motion_id"]): item for item in read_jsonl(args.manifest)}
    source_mean = np.load(args.source_mean).astype(np.float32)
    source_std = np.load(args.source_std).astype(np.float32)
    t2m_mean = np.load(args.t2m_mean).astype(np.float32)
    t2m_std = np.load(args.t2m_std).astype(np.float32)

    mdm_root = args.mdm_root.resolve()
    os.chdir(mdm_root)
    sys.path.insert(0, str(mdm_root))
    import torch
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
    from torch.utils.data import DataLoader

    evaluator = EvaluatorMDMWrapper("humanml", torch.device(args.device))
    vectorizer = WordVectorizer("./glove", "our_vab")
    replication_stats = []
    for file_path in index["files"]:
        with np.load(file_path) as generated:
            features = generated["features"].astype(np.float32)
            ids = [str(value) for value in generated["motion_ids"]]
            lengths = generated["lengths"].astype(int)
        gen_items, gt_items = [], []
        for sample_index, (motion_id, length) in enumerate(zip(ids, lengths)):
            row = manifest[motion_id]
            caption, tokens = first_caption(row["text_path"])
            record = {"caption": caption, "tokens": tokens}
            raw_generated = features[sample_index] * source_std[None] + source_mean[None]
            eval_generated = (raw_generated - t2m_mean[None]) / t2m_std[None]
            raw_gt = np.load(row["joint_vec_path"]).astype(np.float32)
            eval_gt = (raw_gt - t2m_mean[None]) / t2m_std[None]
            gen_items.append(motion_item(record, eval_generated, int(length)))
            gt_items.append(motion_item(record, eval_gt, int(length)))
        loaders = []
        for items in (gt_items, gen_items):
            loaders.append(DataLoader(
                EvalMotionDataset(items, vectorizer), batch_size=args.batch_size,
                collate_fn=collate_eval_batch, drop_last=True, shuffle=False, num_workers=0,
            ))
        replication_stats.append(compute_metrics(
            evaluator, loaders[0], loaders[1],
            diversity_times=min(args.diversity_times, len(gen_items) - 1),
        ))
        print(json.dumps(replication_stats[-1]), flush=True)
    payload = {
        "generation_index": str(args.generation_index.resolve()),
        "mode": index["mode"],
        "sensor_config": index.get("sensor_config"),
        "num_records": index["num_samples"],
        "metrics": aggregate_stats(replication_stats),
        "replication_metrics": replication_stats,
        "multimodality": None,
        "multimodality_note": "not computed: one generation per caption per replication",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
