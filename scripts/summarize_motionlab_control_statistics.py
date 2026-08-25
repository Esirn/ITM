#!/usr/bin/env python3
"""Build paired uncertainty estimates for the MotionLab control experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COMPARISONS = {
    "text": "Text-only",
    "zero": "zero control",
    "shuffled": "shuffled IMU",
}


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_rate_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    return bootstrap_mean_ci(values.astype(np.float64), rng=rng, samples=samples)


def sign_flip_pvalue(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    permutations: int,
) -> float:
    """Two-sided paired randomization test for a non-zero mean difference."""
    observed = abs(float(differences.mean()))
    signs = rng.choice((-1.0, 1.0), size=(permutations, len(differences)))
    permuted = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (permutations + 1))


def summarize_config(
    records: list[dict],
    config: str,
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
    permutations: int,
) -> dict:
    paired = np.asarray(
        [record["configs"][config]["active_error_m"]["paired"] for record in records]
    )
    comparisons = {}
    for key, label in COMPARISONS.items():
        reference = np.asarray(
            [record["configs"][config]["active_error_m"][key] for record in records]
        )
        difference = paired - reference
        improved = difference < 0
        comparisons[key] = {
            "label": label,
            "paired_minus_reference_mean_m": float(difference.mean()),
            "paired_minus_reference_ci95_m": bootstrap_mean_ci(
                difference, rng=rng, samples=bootstrap_samples
            ),
            "improved_count": int(improved.sum()),
            "improved_rate": float(improved.mean()),
            "improved_rate_ci95": bootstrap_rate_ci(
                improved, rng=rng, samples=bootstrap_samples
            ),
            "paired_sign_flip_pvalue": sign_flip_pvalue(
                difference, rng=rng, permutations=permutations
            ),
        }

    text_jerk = np.asarray(
        [record["configs"][config]["text_jerk_ratio"] for record in records]
    )
    paired_jerk = np.asarray(
        [record["configs"][config]["paired_jerk_ratio"] for record in records]
    )
    jerk_difference = paired_jerk - text_jerk
    return {
        "samples": len(records),
        "active_error_comparisons": comparisons,
        "jerk_paired_minus_text_mean": float(jerk_difference.mean()),
        "jerk_paired_minus_text_ci95": bootstrap_mean_ci(
            jerk_difference, rng=rng, samples=bootstrap_samples
        ),
        "jerk_paired_sign_flip_pvalue": sign_flip_pvalue(
            jerk_difference, rng=rng, permutations=permutations
        ),
    }


def render_markdown(payload: dict) -> str:
    def format_pvalue(value: float) -> str:
        return "<0.0001" if value < 0.0001 else f"{value:.4f}"

    lines = [
        "# MotionLab Control Statistical Analysis",
        "",
        (
            f"Paired analysis over the same {payload['sample_count']} test motions. "
            f"Intervals use {payload['bootstrap_samples']} deterministic bootstrap "
            f"resamples; p-values use {payload['permutations']} paired sign-flip "
            "randomizations. Negative error differences favor paired IMU."
        ),
        "",
        "| Sensor | Reference | Error change (m), 95% CI | Improved, 95% CI | p |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for config, summary in payload["configs"].items():
        for item in summary["active_error_comparisons"].values():
            low, high = item["paired_minus_reference_ci95_m"]
            rate_low, rate_high = item["improved_rate_ci95"]
            lines.append(
                f"| {config} | {item['label']} | "
                f"{item['paired_minus_reference_mean_m']:+.4f} [{low:+.4f}, {high:+.4f}] | "
                f"{item['improved_count']}/{summary['samples']} "
                f"({item['improved_rate']:.0%}) [{rate_low:.0%}, {rate_high:.0%}] | "
                f"{format_pvalue(item['paired_sign_flip_pvalue'])} |"
            )
    lines.extend([
        "",
        "| Sensor | Jerk change, 95% CI | p |",
        "| --- | ---: | ---: |",
    ])
    for config, summary in payload["configs"].items():
        low, high = summary["jerk_paired_minus_text_ci95"]
        lines.append(
            f"| {config} | {summary['jerk_paired_minus_text_mean']:+.4f} "
            f"[{low:+.4f}, {high:+.4f}] | "
            f"{format_pvalue(summary['jerk_paired_sign_flip_pvalue'])} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- These are paired diagnostics on the fixed 100-motion subset, not a full HumanML3D benchmark.",
        "- The active-joint trajectory error is a joint-space proxy, not generated IMU orientation error.",
        "- A small p-value for the mean does not imply that control improves every sequence; the improvement rate and its interval report that stability separately.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    records = source["records"]
    rng = np.random.default_rng(args.seed)
    payload = {
        "source": str(args.input.resolve()),
        "sample_count": len(records),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "configs": {
            config: summarize_config(
                records,
                config,
                rng=rng,
                bootstrap_samples=args.bootstrap_samples,
                permutations=args.permutations,
            )
            for config in ("head", "wrists")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
