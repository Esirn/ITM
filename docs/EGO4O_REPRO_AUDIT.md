# Ego4o Reproducibility Audit

## Current Finding

Ego4o code is available at `/home/a200/0relatedworks/ego4o-code-release/`, mainly split into:

- `EgoOmniMocap`: mocap, VQ-VAE, IMUPoser-style encoders, metrics, configs
- `llava`: LLaVA-based motion understanding branch

The code is useful as a reference, but ITM will not make Ego4o reproduction a main workstream.

The local audit currently reports:

- 63 Ego4o config files with hard-coded historical absolute paths.
- 61 unique missing hard-coded paths.
- Missing examples include pretrained TLControl/VQ-VAE weights, IMUPoser root, Nymeria root, generated LLaVA text JSON files, and normalization statistics.
- `/home/a200/0proj/datasets/mdm-need/HumanML3D/texts` is missing, while AMASS, SMPL, and motion split files are present.

## Reproducibility Risks

- `EgoOmniMocap/README.md` is effectively empty.
- Many configs contain hard-coded historical paths:
  - `/CT/EgoMocap/...`
  - `/scratch/inf0/user/jianwang/nymeria`
  - `/home/jianwang/EgoMocap/...`
- Several configs require pretrained VQ-VAE/transformer/checkpoint files whose local availability is not yet verified.
- Nymeria data paths are hard-coded and may not exist locally.
- The LLaVA directory mostly retains upstream LLaVA documentation, so Ego4o-specific training/eval entrypoints need code-level tracing.

## Useful Components

- `mmpose/models/ego_omni_mocap/imuposer_encoder.py`
- `mmpose/models/ego_omni_mocap/vqvae/`
- `mmpose/datasets/datasets/imuposer/`
- `mmpose/datasets/datasets/nymeria/`
- `mmpose/evaluation/ego_omni_mocap_metrics/`
- `configs/imuposer/`
- `configs/nymeria/`
- `configs/ego_omni_mocap/`

## Recommended Use

Use Ego4o primarily for architecture and data-pipeline reference:

- IMU/text/image fusion configuration patterns.
- VQ-VAE motion representation choices.
- Random modality masking.
- Metric and visualization conventions.

Only create local override configs if a specific baseline comparison requires running part of Ego4o.

Recommended command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_assets.py --config configs/paths.toml --max-ego4o-files 8
```
