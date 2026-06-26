# ITM Environment

The project uses the conda environment `itm`.

Installed lightweight base packages:

- Python 3.10
- pip
- numpy
- pytest
- tqdm
- PyYAML

PyTorch and other large training dependencies are optional. The linear baselines
do not require PyTorch.

## Optional PyTorch Dependency

The neural baseline scripts require PyTorch, but the current `itm` environment
does not have `torch` installed. Attempts to install from official PyTorch CUDA
12.1 and CPU wheel indexes stalled on 2026-06-26 and were interrupted.

When a reliable mirror is available, install PyTorch in `itm` and verify:

```bash
conda run -n itm python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use GPU 0 for local neural smoke tests unless GPU availability changes. GPU 1
currently has another long-running process and should be avoided.

## Setup

For an existing environment:

```bash
conda activate itm
pip install -e .
```

For recreation:

```bash
conda env update -n itm -f environment.yml
```

## Verification

```bash
conda run -n itm python -m pytest tests
conda run -n itm python scripts/audit_assets.py --config configs/paths.toml --max-ego4o-files 2
```
