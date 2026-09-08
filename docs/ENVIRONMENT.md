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
- SciPy 1.15.3 and scikit-learn 1.7.2
- OpenAI CLIP, blobfile, spaCy, smplx, h5py, and chumpy for MDM/SMPL baselines

The linear baselines do not require PyTorch. The neural baseline scripts do.

The OpenAI CLIP package used by official MDM was installed from the local
archive to avoid a Git/network dependency:

```bash
conda run -n itm pip install --no-build-isolation \
  /home/a200/0proj/datasets/mdm-need/CLIP-main.zip
```

Legacy SMPL pickle loading requires `chumpy==0.70`. ITM wrappers provide local
NumPy 2.x compatibility aliases; do not downgrade the project-wide NumPy.

Install chumpy separately because its legacy build does not support pip build
isolation:

```bash
conda run -n itm pip install --no-build-isolation chumpy==0.70
```

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

Always inspect GPU utilization before launching a job. Device selection is a
runtime decision; experiment entry points must keep an explicit `--device`
argument and must not terminate unrelated processes.

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

## HY-Motion Environment

HY-Motion requires Qwen3 support and NumPy below 2, which conflicts with the
verified MDM environment. It therefore runs in the separate `hymotion`
environment defined by `environment-hymotion.yml`:

```bash
conda env create -f environment-hymotion.yml
conda run -n hymotion python -c \
  "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"
```

Expected core versions are PyTorch 2.5.1/CUDA 12.1, Transformers 4.53.3 and
NumPy 1.26.4. See `docs/HOST_HYMOTION_HANDOFF_2026-08-26.md` for host paths,
non-Git assets, clean-source extraction and the large-memory training handoff.
