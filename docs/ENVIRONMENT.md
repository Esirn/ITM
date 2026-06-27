# ITM Environment

The project uses the conda environment `itm`.

Installed lightweight base packages:

- Python 3.10
- pip
- numpy
- pytest
- tqdm
- PyYAML
- PyTorch 2.5.1 with CUDA 12.1
- Transformers 4.49.0

The linear baselines do not require PyTorch. The neural baseline scripts do.

## PyTorch Dependency

PyTorch was installed with:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

If importing torch fails with
`undefined symbol: iJIT_NotifyEvent`, downgrade MKL/OpenMP in the environment:

```bash
conda install -n itm -y "mkl<2025" "intel-openmp<2025"
```

The verified working package combination is:

- `pytorch 2.5.1 py3.10_cuda12.1_cudnn9.1.0_0`
- `pytorch-cuda 12.1`
- `mkl 2023.1.0`
- `intel-openmp 2023.0.0`
- `sympy 1.13.1`
- `transformers 4.49.0`

Verify with:

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
