# ITM Environment

The project uses the conda environment `itm`.

Installed lightweight base packages:

- Python 3.10
- pip
- numpy
- pytest
- tqdm
- PyYAML

PyTorch and other large training dependencies are intentionally not installed yet. Choose them later based on the target CUDA version and the selected text-to-motion backbone.

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

