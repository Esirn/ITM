#!/usr/bin/env python
"""Cache frozen Transformer caption embeddings for one manifest split."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import numpy as np

from itm.data.dataset import TextIMUMotionDataset
from itm.data.text_embeddings import save_text_embedding_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument(
        "--encoder",
        choices=("auto", "mean-pool", "clip"),
        default="auto",
        help="Text pooling protocol. 'auto' detects CLIP from the model config.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    dataset = TextIMUMotionDataset(
        args.manifest, load_joints=False, load_joint_vec=False, load_imu=False
    )
    samples = list(islice((dataset[index] for index in range(len(dataset))), args.max_records))
    motion_ids = [str(sample["motion_id"]) for sample in samples]
    captions = [str(sample["caption"]) for sample in samples]
    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=local_only
    )
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=local_only)
    encoder = args.encoder
    if encoder == "auto":
        encoder = "clip" if model_config.model_type == "clip" else "mean-pool"
    if encoder == "clip":
        from transformers import CLIPTextModel

        model = CLIPTextModel.from_pretrained(args.model, local_files_only=local_only)
    else:
        model = AutoModel.from_pretrained(args.model, local_files_only=local_only)
    device = torch.device(args.device)
    model.to(device).eval()

    encoded_batches = []
    with torch.inference_mode():
        for start in range(0, len(captions), args.batch_size):
            tokens = tokenizer(
                captions[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            output = model(**tokens)
            if encoder == "clip":
                pooled = output.pooler_output
            else:
                token_mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (output.last_hidden_state * token_mask).sum(1)
                pooled = pooled / token_mask.sum(1).clamp_min(1)
            encoded_batches.append(pooled.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(encoded_batches) if encoded_batches else np.empty((0, 0))
    save_text_embedding_cache(
        args.output,
        motion_ids,
        captions,
        embeddings,
        metadata={
            "model": args.model,
            "encoder": encoder,
            "pooling": "clip_eos_pooler" if encoder == "clip" else "attention_mask_mean",
            "max_length": args.max_length,
            "manifest": str(Path(args.manifest)),
        },
    )
    print(f"records: {len(samples)}")
    print(f"embedding_dim: {embeddings.shape[1]}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
