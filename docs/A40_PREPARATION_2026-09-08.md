# A40 host preparation (2026-09-08)

## Scope

Initial migration preparation used CPU only. The subsequently authorized
CUDA Text-only smoke and later controlled-training experiments are recorded
below; at the original preparation checkpoint, no GPU training had been launched.
At 15:41 CST, all four A40 GPUs had 97-100% utilization and approximately
28-37 GiB allocated. Query GPU availability again immediately before any run.
The sandbox's initial NVIDIA driver error was a visibility restriction;
the host query succeeded with driver 570.133.07.

ITM started on `master`, commit `3523bbc`, with a clean worktree. A read-only
`git ls-remote origin refs/heads/master` confirmed the same full commit,
`3523bbcf4f96cb3c2c3cc21b8c1aec125e548ff6`. The workspace resolves to
`/home/4T-2/syx0011/0proj/ITM`.

## Existing environment checks

Both environments were inspected without modifying them. Checks used
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=2`, offline Hugging Face access,
and the extracted clean runtime on `PYTHONPATH`.

| Check | syx0011 `hymotion-syx` | wcy0170 `hymotion` |
|---|---|---|
| Python environment | `/home/syx0011/miniconda3/envs/hymotion-syx` | `/home/wcy0170/anaconda3/envs/hymotion` |
| PyTorch | 2.5.1+cu124 | 2.5.1+cu124 |
| NumPy | 2.2.5 | 1.26.4 |
| Transformers | 4.53.3 | 4.45.2 |
| NumPy/Torch CPU round trip | Pass | Pass |
| Clean HYTextModel and MotionFlowMatching imports | Pass | Pass |
| Local Qwen3 AutoConfig | Pass | Fails: unrecognized model type `qwen3` |
| Local Qwen3 tokenizer | Pass | Not reached |
| Local CLIP AutoConfig | Pass | Not reached |

The user's observation that Conda lists NumPy 1.26.4 is correct. Follow-up
inspection found **both** `numpy-1.26.4.dist-info` and
`numpy-2.2.5.dist-info`, plus Conda `numpy-base 2.2.5`, in `hymotion-syx`.
Actual `import numpy` reports 2.2.5 from that environment's `site-packages`.
Conda's listed version alone is therefore insufficient for this environment.

Both original environments use CUDA 12.4 PyTorch builds,
whereas the original reproduction used CUDA 12.1. Record this difference in
future experiment metadata; do not claim an identical software stack.

Neither original environment has pytest, tomli, or torchaudio installed. No
Conda environment named `itm` exists. The requested new `hymotion` Conda creation was
interrupted at approval and its destination directory does not exist.

### Project-local runtimes

Two venvs now reuse the original `hymotion-syx` system packages (including
PyTorch), with project-local overrides. No original Conda environment was
modified. These are **venvs, not Conda environments**:

| Interpreter, relative to ITM | Actual NumPy | Actual Transformers |
|---|---|---|
| `outputs/envs/hymotion/bin/python` | 1.26.4, local | 4.53.3, inherited |
| `outputs/envs/itm/bin/python` | 2.2.5, local | 4.49.0, local |

Both have ITM installed editable. The ITM venv also has pytest, tomli,
chumpy 0.70, OpenCV headless 4.11.0.86, smplx, blobfile, h5py, Matplotlib, and OpenAI
CLIP installed from the handoff's local ZIP. Both pass `pip check`.
The parent `hymotion-syx` environment must remain available; these lightweight
venvs are not self-contained copies. Other inherited package versions differ
from the 4090 environment. Actual import paths/versions and pip inventories
are saved in `outputs/a40_audit/{itm,hymotion}_environment.json`.

An optional dependency defect remains: standalone `import diffusers` fails
because diffusers 0.26.3 imports `cached_download`, removed from the installed
Hugging Face Hub. The official HY-Motion pipeline does not use diffusers;
its only source reference is a loader self-test under `__main__`. This was
recorded without changing the handoff's pinned dependency versions.

CPU pytest after the interpreter integration and cache-auditor tests:
**102 passed, 9 existing Transformer performance warnings**. JUnit report:
`outputs/a40_audit/pytest.xml`.

The unified `HYMotionBackbone` now accepts `python_executable=Path(...)` for
these venvs. Omitting it preserves the existing `conda_env` launch behavior.
Its subprocess test runs a dummy sampler on CPU; it is not a model smoke.

## Source and assets

- External HY-Motion commit: `9993ccd8aeba1f2d34c1061cc2282d14d5193055`, matching the handoff.
- Lite checkpoint SHA-256: `d83f118f8d74db76249db86dcf9982a8229f43ef4e9fa11f683019d6230dd486`, matching the handoff.
- Clean runtime extracted to `outputs/hymotion/runtime_root` using the existing preparation script.
- The external text encoder differs from the committed version. Experiments must use the clean runtime.
- Source/text encoder hashes are recorded in `outputs/hymotion/runtime_root/workspace.json`.
- All five Qwen safetensors shard headers match all 399 tensor entries in the index (`outputs/a40_audit/qwen_shards.json`). This is not a full tensor-content checksum.
- HY-Motion stats, Lite config, CLIP directory, and WoodenMesh assets are present.
- AMASS, SMPL, and the local OpenAI CLIP ZIP paths from the handoff exist.
- The external repository was not modified. A broad diff hit a Git LFS read-only error; source validation used committed Git objects and the text encoder byte comparison.
- Lite's 460,851,401 pipeline parameters load on CPU without unexpected keys or nonfinite tensors. Five missing `body_model.*` buffers come from WoodenMesh assets. No text encoder was loaded and no motion was generated in this checkpoint check (`lite_cpu_checkpoint.json`).
- Local CLIPTextModel CPU encoding produces finite `[1,768]` pooler output (`clip_cpu.json`).
- Clean HYTextModel fully loads Qwen3 and CLIP on CPU and encodes the handoff prompt with finite, nonzero context. Output shapes are `[1,1,768]`, `[1,128,4096]`, `[1]`; context length is 10 (`text_encoder_cpu.json`). No zero-context workaround is used.

### Complete CPU Text-only smoke

The unmodified official pipeline completed a 60-frame, seed-1234 Text-only
sample on **CPU**, including real Qwen3 and CLIP encoding. Core sampling
took 106.78 seconds with four OpenMP/MKL threads. Output is in
`outputs/hymotion/text_only_cpu_smoke/walk_swing_60.npz` and its JSON sidecar.
Finite-value checks passed for motion `[1,60,201]`, local rot6d
`[1,60,22,6]`, global rotations `[1,60,22,3,3]`, and joints `[1,60,52,3]`.
Head and wrists joint-mounted virtual IMU exports also passed finite checks.
See `outputs/a40_audit/cpu_smoke_shapes.json`.
The 60-frame GIF is `outputs/hymotion/text_only_cpu_smoke/walk_swing_60.gif`;
frames 0/15/30/45/59 were inspected in `walk_swing_frames.png`. The skeleton
is visible with changing leg and arm poses. This single sample does not
establish semantic quality across prompts or seeds.

This is a new CPU reference, not a CUDA benchmark or numerical equivalence
test against the 4090 artifacts. The subsequent CUDA check is recorded below.
At this point on September 8, no adapter training or backward smoke had run.
See the September 9 update below for subsequent work and coordinate corrections.

### CUDA Text-only smoke

With user authorization, the same 60-frame prompt and seed 1234 were run on
`cuda:0`. Immediately before launch, GPU 0 had 28,482 MiB free and 30% GPU
utilization from another workload. The official clean pipeline completed
without OOM. Core sampling took **6.3259 seconds**; peak PyTorch allocated
memory was **19,593,442,304 bytes (18.25 GiB)**. This metric is not total
driver allocation and excludes other processes; timing is from a shared GPU.

Motion `[1,60,201]`, local rotations `[1,60,22,6]`, global rotations
`[1,60,22,3,3]`, and head/wrists virtual IMU exports passed finite-value and
shape checks. Global rotation orthogonality error was at most 1.16e-5.
Artifacts are in `outputs/hymotion/text_only_smoke/`, including
`walk_swing_60.npz`, the JSON sidecar, `walk_swing_60.gif`, and both IMU exports.
The validation report is `outputs/a40_audit/cuda_smoke.json`.

The CUDA process exited normally. This validates Text-only inference under
the observed memory availability, not adapter training memory requirements.

### Standard IMU dependencies

The missing [official IMUPoser source](https://github.com/FIGLAB/IMUPoser) was
cloned into `outputs/deps/IMUPoser`, commit
`a0e9d33b9c57d6737d711185b62084e17684b0e2`. Its bundled SMPL male 1.0.0
asset is byte-identical to the host asset:
SHA-256 `0e8c0bbbbc635dcb166ed29c303fb4bef16ea5f623e5a89263495a9e403575bd`.
The original 4090 IMUPoser source revision was not supplied, so byte-for-byte
equivalence of regenerated caches to the old caches is not yet established.

## Host data paths

Use `--config configs/paths.a40.toml` for scripts accepting a path config.
This is a focused HY-Motion/data/MDM asset configuration, not an audit of every related
baseline repository. The original `configs/paths.toml` still describes the
4090 machine and must not be used by default on this host.

The complete HumanML3D root contains every required text, joint, and joint
vector file for all 29,228 standard split entries:

| Split | Entries |
|---|---:|
| train | 23,384 |
| val | 1,460 |
| test | 4,384 |

Manifests were rebuilt into `outputs/manifests_full/{train,val,test}.jsonl`
with host absolute paths. No HumanML motion arrays or HY-Motion weights were copied.
These are HumanML manifests. All AMASS-backed records have existing source
files: 7009 train, 434 val, 1333 test. All 8776 standard IMU caches were rebuilt
at 30 FPS into `outputs/standard_imu`, using the existing builder
with `--device cpu`, two OpenMP/MKL threads per process, and no GPU visibility.

The first 16 cache samples passed finite-value and six-sensor shape checks.
The complete train/val/test caches passed the combined audit in
`outputs/a40_audit/standard_imu.json` (2,467,056,892 bytes total).

| Split | Cache records | Eligible for HY-Motion's 20-360 frame range |
|---|---:|---:|
| train | 7009 | 7003 |
| val | 434 | 434 |
| test | 1333 | 1331 |

Six train records and two test records are shorter than 20 frames. IDs and
lengths are recorded in `outputs/hymotion/data_eligibility.json`. Preserve
them for the original baseline protocol; apply an explicit length policy
when implementing HY-Motion training. No training subset was silently changed.
The cache auditor checks complete AMASS ID coverage, source paths, split
membership, metadata, shapes, finite values, and valid rotation matrices.
Maximum rotation-matrix validation error across splits is below 1.5e-6.

Training acceleration normalization was recomputed using the original MDM
normalization routine and compared with the transferred Stage-2b checkpoint:
maximum absolute mean difference 7.97e-8, standard-deviation difference
4.77e-7. This supports consistency of regeneration but is not a per-file
comparison against the original caches. See
`outputs/a40_audit/imu_normalization_comparison.json` and `cache_provenance.json`.

Do not confuse the sensor definitions: standard IMU wrist orientation uses
SMPL joints 18/19 and vertex-mounted acceleration, whereas the current
HY-Motion exporter uses joints 20/21 and joint-mounted acceleration. Their
coordinate/mounting correspondence still needs explicit treatment before
cross-backbone IMU error comparisons. No such conversion was performed here.

MDM statistics remain separate from the complete data root. MDM commands
must pass explicit `--mean` and `--std`; the new `[mdm_stats]` section records
their paths but is not automatically consumed by those commands.
Observed SHA-256 values (no original files available for a direct comparison):

- Mean.npy: `26e136555dab04c94a129d446c26e6b9939cbf045fbf77bcf5462c1fb5a2001c`
- Std.npy: `6565a65ed9b31e23c328829a309e1c482be8b85fd23b43d65451a9b19a917f40`

## Transferred MDM checkpoints

The user transferred both files to `outputs/mdm_control/`. Both passed
`torch.load(..., map_location="cpu", weights_only=True)`, strict loading of
the IMU encoder and eight residual adapters, and finite parameter checks.

| File | Epoch | Bytes | SHA-256 |
|---|---:|---:|---|
| `stage1_full_pilot_v2.pt` | 5 | 101045130 | `62f789fe3fb405390f4ee71ccb544465debfb625ec162167564de4c8e9506b71` |
| `stage2b_balanced_b.pt` | 8 | 101045290 | `5d0800e61994c79f862378871e2c149fde052c7419553939a06a6a6632788a98` |

Both have six 12D sensor slots, latent dimension 512, head slot 4 and wrist
slots 0/1. Original checkpoint bytes were preserved. Their embedded
`mdm_checkpoint` values still point to the 4090 machine. MDM commands must
override `--mdm-checkpoint`, `--mdm-args`, and `--mdm-root` with the validated
paths in `configs/paths.a40.toml`, in addition to explicit `--mean/--std`.

The shared MDM `save/humanml_trans_enc_512/model000475000.pt` and `args.json`
were found and are byte-identical to the matching members of the shared
`humanml_trans_enc_512.zip` archive. Base checkpoint SHA-256:
`ce43c1d4232fc2b897999d7996c2d2b889791d99585d87e1289af773ab56ab93`.
Reports: `outputs/a40_audit/mdm_checkpoints.json` and `mdm_base_assets.json`.
Full MDM sampling/evaluation has not been rerun.

## Remaining gates

1. Address the documented sensor-definition correspondence and short-sequence policy when preparing HY-Motion training targets.
2. Complete controlled prescreen evaluation and semantic review; see September 9 update below.

The A40 host now has both CPU and CUDA Text-only smoke results. Quantitative
agreement with the original 4090 artifacts has not been tested.

## Commands for this host

Run from the ITM root. Use these explicit interpreters instead of
`conda run -n itm` or `conda run -n hymotion`:

```bash
ITM_PY="$PWD/outputs/envs/itm/bin/python"
HY_PY="$PWD/outputs/envs/hymotion/bin/python"
HYMOTION_ROOT=/home/4T-2/group_motion/relatedworks/HY-Motion-1.0

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$ITM_PY" -m pytest -q

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 \
  "$ITM_PY" scripts/audit_standard_imu_cache.py \
  --config configs/paths.a40.toml --output outputs/a40_audit/standard_imu.json
```

Cache regeneration (existing complete files are reused by the builder):

```bash
for split in train val test; do
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$ITM_PY" scripts/build_standard_imu_cache.py \
    --manifest "outputs/manifests_full/$split.jsonl" \
    --index /home/4T-2/group_motion/relatedworks/HumanML3D/index.csv \
    --amass-root /home/4T-2/group_motion/datasets/AMASS \
    --imuposer-root outputs/deps/IMUPoser \
    --output-dir outputs/standard_imu \
    --cache-manifest "outputs/manifests_full/${split}_standard_imu.jsonl" \
    --target-fps 30 --device cpu
done
```

After the user identifies an available GPU and a fresh `nvidia-smi` confirms
availability, set `DEVICE` to that device, then run:

```bash
"$HY_PY" scripts/sample_hymotion_text.py \
  --hymotion-root "$HYMOTION_ROOT" \
  --runtime-root outputs/hymotion/runtime_root --model lite \
  --text "a person walks forward while swinging both arms" \
  --frames 60 --seed 1234 --device "$DEVICE" \
  --output outputs/hymotion/text_only_smoke/walk_swing_60.npz
```

For the Python backbone API:

```python
from pathlib import Path
from itm.backbones import HYMotionBackbone

backbone = HYMotionBackbone(
    repo_root=Path("/home/4T-2/group_motion/relatedworks/HY-Motion-1.0"),
    runtime_root=Path("outputs/hymotion/runtime_root"),
    python_executable=Path("outputs/envs/hymotion/bin/python"),
    device=device,  # Set to the available GPU at launch time.
)
```

## September 9: Control Implementation and Prescreen

User designated `cuda:1`; observed 28,549 MiB free and 0% GPU utilization,
with memory still occupied by other processes. No other processes were stopped.

Two coordinate defects were found during official WoodenMesh comparison:

- Official 6D rotations store the first two columns interleaved, not two rows.
  Earlier ITM `global_rotations_mat` and exported orientations were incorrect.
- Official decoded `keypoints3d` omit root translation. World joints must be
  reconstructed from rotations and ground-aligned `transl`, not by adding
  translation to already ground-shifted official joints (which double-shifts Y).

The repaired result is
`outputs/hymotion/text_only_smoke/walk_swing_60_world_v2.npz`, with matching
`world_v2_head_virtual_imu.npz`, `world_v2_wrists_virtual_imu.npz`, and GIF.
Original outputs are preserved for provenance and must not be used for IMU
evaluation. The intermediate `walk_swing_60_world.npz` is also invalid.
Maximum world-FK discrepancy against official WoodenMesh: 7.1525574e-7 m.
New exports reject results without verified world/rotation metadata.

Implemented `src/itm/models/hymotion_imu_adapter.py`: six-slot 12D sensor
attention, temporal encoder, zero-initialized 1024D residual projections into
6 double-stream motion branches and 12 single-stream motion prefixes. Text
tokens and the 460,841,673-parameter backbone stay frozen; 21,530,112 adapter
parameters train. Masks and independent text/IMU dropout are supported.

`scripts/train_hymotion_imu.py` builds a separate WoodenMesh joint-mounted
representation from real standard-cache AMASS rotations/translations and
full-clip HumanML captions. It does not modify MDM caches. Initial facing is +Z,
root XZ starts at the origin, and mesh minimum Y is on the floor. The last 66D
are world-axis root-relative joints, an explicitly documented ITM reconstruction
of the paper representation, not a claim of exact official preprocessing.
Real Qwen/CLIP embeddings are cached before unloading the text encoder.

The first smoke was interrupted before training after finding the 6D defect;
`outputs/hymotion/backward_smoke_20260909` is invalid. Corrected 16-record smoke:
`outputs/hymotion/backward_smoke_20260909_v2`, 16 finite backward steps,
3.757 s training time and 2,574,883,328-byte peak PyTorch allocation after
unloading text encoders (not total process memory or end-to-end preparation).

Prescreen training: `outputs/hymotion/prescreen_512_20260909`, 512 records,
2 epochs, batch size 1, seed 1234, AdamW lr 1e-4, 1024 finite steps.
Epoch mean flow losses: 0.2858364 and 0.2133680. Text dropped on 92 steps;
IMU dropped on 103. Training time 103.808 s, peak allocation 2,570,676,736 bytes.
Adapter SHA-256:
`eed33fea289247fec015d5a1fb0fbe01a4d34e32c1cf08a9996d40e5751e613f`.
These training losses do not establish successful IMU conditioning.

128 held-out validation records and real text embeddings are prepared in
`outputs/hymotion/prescreen_val128_20260909`; no validation training occurred.
Evaluation script `scripts/evaluate_hymotion_imu.py` fixes text, frame count,
seed/noise, 50 Euler steps and text CFG=5 across Text-only/paired/zero/shuffled,
for both head and wrists. It uses training acceleration normalization only.
Results are written incrementally to `outputs/hymotion/prescreen_eval128_20260909`.
Semantic preservation is not inferred from IMU error. Full training stays gated.

Tests after implementation: 110 passed, 9 existing Transformer warnings.
All 640 reconstructed motion/text caches passed
`scripts/audit_hymotion_control_data.py`: maximum IMU round-trip difference
0.0005149841 and root-relative joint difference 9.536743e-7 m. Report:
`outputs/a40_audit/hymotion_control_data.json`.

User subsequently authorized `cuda:2` for evaluation. Fixed global indices
0..63 belong to the first output directory on GPU 1; 64..127 belong to
`outputs/hymotion/prescreen_eval128_20260909_gpu2` on GPU 2. Seeds and shuffled
partners are still defined over all 128 records, not independently per shard.
Any GPU 1 outputs beyond its assigned range are ignored, regardless of scores.
`scripts/merge_hymotion_evaluation.py` audits complete eight-way coverage,
disjoint train/validation IDs, seeds/partners, finite outputs, valid rotations,
and identical Text-only generations across the two sensor configurations.

## Formal Pad-360 Prescreen (In Progress)

The variable-length pilot was isolated as exploratory because the official
single-stream RoPE positions include the motion prefix. A direct real-backbone
check measured maximum output difference 1.675408 when replacing its 360-frame
prefix with a short prefix. The formal adapter was therefore retrained with
360-frame padding and valid-frame-only flow loss:
`outputs/hymotion/prescreen_512_pad360_20260909`. It completed 1,024 finite
steps; epoch mean losses were 0.3139921 and 0.2052559. Peak PyTorch allocation
was 2,638,223,360 bytes. Adapter SHA-256:
`572682c993e3b8234ff743e76b3de04497a16114cee210d68656dff3bb3a2bdb`.

Formal validation indices 0..63 completed on GPU 1 in
`outputs/hymotion/prescreen_pad360_eval_gpu1_20260909`. Real-backbone checks
were exact at zero initialization and after placing all 18 immutable RoPE
tables on GPU. Peak allocation was 6,209,158,144 bytes. Paired beat shuffled
in 47/64 head cases (73.44%) and 51/64 wrist cases (79.69%), but paired mean
jerk was 7.54x Text-only for head and 6.09x for wrists. This fails the required
non-worsening jerk gate; it is not authorization for full 7,009-record
training. The remaining indices 64..127 await an actually idle GPU 2.

CPU visual review samples for arm rotations, walking-to-dancing, and throw/catch
are saved beside that run. They support only a limited qualitative check; the
formal semantic gate remains unpassed until the complete held-out result has
been reviewed.

### Complete Formal Validation

The remaining fixed indices 64..127 completed on GPU 3 without OOM. The formal
merge/audit is `outputs/hymotion/prescreen_pad360_eval128_20260909/summary.json`.
It checked all 128 validation records and 1,024 generated outputs for complete
coverage, train/validation disjointness, fixed seed and cyclic shuffled partner,
finite arrays/metrics, output-metadata agreement, metric recomputation, and
SO(3) validity. Maximum rotation orthogonality error: 1.5497208e-6.

Paired beat shuffled in 95/128 head cases (74.22%) and 105/128 wrist cases
(82.03%), so the paired-vs-shuffled gate passes. The mandatory jerk gate fails:
paired mean jerk is 327.2454 versus 48.0099 m/s^3 Text-only for head (6.82x),
and 274.1077 versus 48.0099 m/s^3 for wrists (5.71x). The formal summary sets
`full_training_allowed: false`. Do not start the 7,009-record full training
with this adapter/checkpoint. The likely next experiment is a revised objective
or regularization that explicitly controls generated acceleration/jerk, followed
by a fresh prescreen rather than continuing the failed run.

### Inference-Strength Diagnostic

The adapter now exposes a validated inference-only `control_strength` multiplier
on its IMU residual; `0` is exactly the unconditional backbone, and `1` is the
formal checkpoint behavior. This is a diagnostic, not a replacement for a new
training objective. On the same fixed first 16 validation records, seed/noise,
and four-way paired/zero/shuffled protocol, strength `0.25` reduced paired jerk
to 1.547x Text-only (head) and 1.508x (wrists), but head paired-vs-shuffled was
only 68.75%. Strength `0.10` further reduced jerk to 1.105x and 1.075x,
respectively, while head remained 68.75% and wrists were 75.00%. At `0.05`,
jerk reached 1.039x and 1.024x, but head discrimination collapsed to 56.25%
(wrists: 81.25%). Thus this scan excludes inference-only residual scaling as a
solution: reducing jerk toward Text-only also loses the required head IMU
discrimination. Artifacts:
`outputs/hymotion/control_strength_025_eval16_20260909`,
`outputs/hymotion/control_strength_010_eval16_20260909`, and
`outputs/hymotion/control_strength_005_eval16_20260909`.

`scripts/train_hymotion_imu.py` now has the opt-in
`--position-jerk-loss-weight` auxiliary objective for the next prescreen. It
converts the predicted flow endpoint and ground-truth endpoint back to physical
world-axis root-relative joint positions, applies third temporal differences at
30 Hz over valid frames only, and matches those jerks. Its default is `0`, so
it does not alter the formal run or any ordinary smoke. The run metadata and
per-step JSONL record both the weight and unweighted jerk loss. A direct smoke
verified zero loss for an exact endpoint, positive loss for a changed valid
frame, and exclusion of padded frames. The available HY-Motion environments do
not currently include `pytest`, so this change has compilation and direct
numeric smoke coverage but no rerun of the full pytest suite.

A calibrated 512-record, 2-epoch pad-360 prescreen was trained at
`--position-jerk-loss-weight 1e-7` in
`outputs/hymotion/prescreen_512_pad360_jerk1e7_20260909`. The calibration
median had flow loss 0.8318 and raw jerk loss 1.989e6, so this conservative
weight contributed about 0.20 at the median. The run was finite for 1,024
steps (epoch mean flow losses 0.31049 and 0.19906; jerk losses 4.686e5 and
1.298e5), with a 2,635,947,008-byte peak PyTorch allocation. Its fixed 16-case
generation screen passed paired-vs-shuffled discrimination (head 81.25%, wrists
87.50%) but failed jerk badly: paired 254.07 versus 37.72 m/s^3 Text-only for
head (6.74x), and 215.02 versus 37.72 for wrists (5.70x). See
`outputs/hymotion/prescreen_512_pad360_jerk1e7_eval16_20260909`. Do not run
the 128-case formal validation or full training for this configuration.

### September 10 Sampling Diagnostics

`cuda:0` had 31.4 GiB free and was used for a new fixed-16 sampling diagnostic.
`temporal_velocity_smoothing` is an inference-only, motion-only three-point
filter applied to every DiT velocity prediction; its default `0` preserves the
former output exactly, and it applies uniformly to all four evaluation modes.
At smoothing `0.10`, it made jerk substantially worse: paired mean 446.11
versus 80.68 m/s^3 Text-only for head, and 365.99 versus 80.68 for wrists,
despite paired-vs-shuffled fractions of 75.00% and 93.75%. Do not scan that
sampler filter further. Output:
`outputs/hymotion/velocity_smoothing_010_eval16_20260910`.

CPU postprocessing diagnostics on the original first 64 formal outputs isolated
root translation and local rotations without spending GPU time. Repeated
root-only filtering preserved discrimination but plateaued at paired/Text-only
jerk ratios of 2.53x head and 1.95x wrists after 32 iterations. A fixed-bone
length FK reconstruction with both local-6D rotation and root filtering reached
only 4.34x and 3.19x after four iterations. The smoother therefore cannot pass
the relative jerk gate; it is not integrated as a production output transform.
Reports: `outputs/hymotion/root_smoothing_gpu1_64_20260910.json`,
`outputs/hymotion/root_smoothing_gpu1_64_strong_20260910.json`, and
`outputs/hymotion/local_rotation_smoothing_gpu1_64_20260910.json`.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  outputs/envs/hymotion/bin/python scripts/evaluate_hymotion_imu.py \
  --training-run outputs/hymotion/prescreen_512_20260909 \
  --data-run outputs/hymotion/prescreen_val128_20260909 \
  --output outputs/hymotion/prescreen_eval128_20260909 \
  --device cuda:1 --count 128
```

Use a new output directory for reruns; existing runs are never overwritten.

## September 20: Further Jerk and Conditioning Prescreens

The following 512-record, two-epoch pad-360 screens all completed without
numerical or memory failures. They are screening results only: none qualifies
for a 128-record formal validation or 7,009-record full training.

An Euler two-step unrolled position-jerk objective (weight `2e-8`) was finite
for 1,024 steps, but its fixed-16 screen had paired jerk 330.0 m/s^3 (head)
and 378.3 m/s^3 (wrists), respectively 8.75x and 10.03x Text-only. Its
paired-vs-shuffled fractions were 68.75% and 81.25%. This rules out that
particular unrolled-loss calibration.

Low-pass control-token experiments were more useful but still missed the jerk
gate. With stride 4 token pooling/interpolation, paired jerk was 155.62
(head) and 158.54 (wrists) m/s^3, while discrimination was 87.50% and
100.00%. Stride 8 reached 152.72 and 142.23 m/s^3 with 75.00% and 87.50%
discrimination. Adding mask-preserving zero-IMU dropout to stride 8 made the
screen worse (175.48 and 142.66 m/s^3; 75.00% and 87.50%). Thus smoother,
lower-rate control tokens reduce but do not eliminate the generated-motion
jerk; no variant should be promoted unchanged. Relevant outputs use the
`prescreen_512_pad360_*stride*` naming family under `outputs/hymotion`.

`condition_representation=anchors` was then implemented and screened to test
a lower-derivative condition: each sensor token contains root-relative sensor
joint position plus the original global orientation (still 6x12). The adapter
uses stride-4 token pooling. The implementation trained successfully for 1,024
steps at `outputs/hymotion/prescreen_512_pad360_anchors_stride4_20260920`
(adapter SHA-256 `f2fe6623250864dfaa8ce6067009049cd0ad1226b5ebfb1517dc99ed30cbe281`).
Its fixed-16 four-way evaluation is
`outputs/hymotion/prescreen_512_pad360_anchors_stride4_eval16_20260920`.
It fails both selection criteria: paired-vs-shuffled fraction was 50.00% for
head and 68.75% for wrists, and paired jerk was 166.77 and 146.02 m/s^3,
respectively 4.42x and 3.87x the 37.72 m/s^3 Text-only baseline. Do not extend
the anchor representation to formal validation or full training.

The next experiment must change the learned temporal motion mechanism rather
than add another inference smoother or reweight the same endpoint objective.
Keep all prior failed configurations as negative controls, use a fixed-16
screen first, and require both paired-vs-shuffled discrimination and the
relative jerk gate before spending A40 time on the 128-record or full run.

### Late-Block Control Injection

`HYMotionIMUConfig.control_injection_start` was added with default `0`, which
preserves the all-18-block behavior. It disables residual injection (and its
gradients) before the selected block while retaining the original adapter
architecture. A direct CPU smoke verified that skipped layers receive no
gradient and the remaining layer still controls the motion prefix.

The first altered temporal mechanism used stride-4 tokens but injected only
into blocks 12--17 (the final six blocks). It trained for 1,024 finite steps at
`outputs/hymotion/prescreen_512_pad360_stride4_late6_20260920` (adapter
SHA-256 `d3bcffc0c7634942328bdfb83737da0f4c9135fbe7c855bea6a3962de3a8a7d2`).
The fixed-16 evaluation at
`outputs/hymotion/prescreen_512_pad360_stride4_late6_eval16_20260920` retained
discrimination (87.50% head, 81.25% wrists) but failed jerk more strongly than
the all-block stride-4 baseline: paired jerk was 192.06 m/s^3 head (5.09x
Text-only) and 178.61 m/s^3 wrists (4.73x Text-only). Do not extend this
late-only configuration to formal validation or full training.

The complementary first-six-block window was substantially smoother. It trained
for 1,024 finite steps at
`outputs/hymotion/prescreen_512_pad360_stride4_early6_20260920` (adapter
SHA-256 `14c55c770068da963b8cd09bc54ff8e324ae69ed207b31f6690d57222193bba7`).
Its fixed-16 output is
`outputs/hymotion/prescreen_512_pad360_stride4_early6_eval16_20260920`:
paired-vs-shuffled was 87.50% head and 81.25% wrists, while paired jerk fell to
90.51 and 87.34 m/s^3, respectively 2.40x and 2.32x Text-only. This is the
best jerk/discrimination trade-off so far but still fails the relative jerk
gate, so it must not be promoted to 128 records or full training. The next
bounded screen is the narrower first-three-block window, not a repeat of any
failed full/late/anchor configuration.

The first-three-block window trained for 1,024 finite steps at
`outputs/hymotion/prescreen_512_pad360_stride4_early3_20260920` (adapter
SHA-256 `dc69c7cccb1fb37fdf41a3618ab1e4b52a95b653189fdde58a27bb5d6a7dae7a`).
It improved paired jerk again to 82.36 m/s^3 head and 81.88 m/s^3 wrists
(2.18x and 2.17x Text-only), but weakened paired-vs-shuffled fractions to
81.25% and 75.00%. The fixed-16 output is
`outputs/hymotion/prescreen_512_pad360_stride4_early3_eval16_20260920`.
It remains ineligible for formal validation because the relative jerk gate is
not met. The only remaining informative layer-window boundary test is first
block only; if it does not pass both gates, stop scanning injection depth.

The first-block-only boundary test also trained for 1,024 finite steps at
`outputs/hymotion/prescreen_512_pad360_stride4_early1_20260920` (adapter
SHA-256 `243190872184a18604cc643c7a0243a40e8a4e6677f0d93d9f1e3c378f7bb155`).
Its fixed-16 evaluation at
`outputs/hymotion/prescreen_512_pad360_stride4_early1_eval16_20260920` further
reduced paired jerk to 62.43 m/s^3 head (1.65x Text-only) and 67.99 m/s^3
wrists (1.80x), but paired-vs-shuffled discrimination collapsed to 62.50% and
68.75%. This fails the discrimination gate, so no 128-record validation or
full training may run. Injection-depth scanning is complete: fewer active
blocks trade condition discrimination for lower jerk, while the other endpoint
of the scan (late-six) increased jerk. Further work must introduce a different
training/control mechanism rather than test another layer-window size.

## September 21: Direct Sensor Supervision Prescreens

Two bounded 512-record, two-epoch experiments tested paired-IMU auxiliary
objectives on the stride-4, first-three-block configuration. Both used 10%
active-zero conditioning in addition to the existing 10% missing-IMU dropout.
The active-zero condition retains the sensor mask but zeros the normalized IMU;
it is therefore distinct from Text-only/missing IMU during training and
evaluation.

The first experiment added a differentiable Smooth-L1 sensor-acceleration loss
with weight `0.01`, calibrated to approximately 10% of the median flow loss.
The physical acceleration implementation matches cached training IMU to mean
absolute error `1.59e-5 m/s^2` over 16 records. Training completed 1,024 finite
steps at
`outputs/hymotion/prescreen512x2_stride4_early3_accel001_20260921` (adapter
SHA-256 `215c11db4d3b9d57fde408dc9c3746e57018452bf016a17f89b83854856da8c9`).
Its fixed-16 evaluation is
`outputs/hymotion/prescreen512x2_stride4_early3_accel001_eval16_20260921`.
Head/wrists paired-vs-shuffled fractions fell to 75.00%/68.75%, while paired
jerk increased to 125.03/121.11 m/s^3, or 3.31x/3.21x Text-only. This objective
is rejected.

That result also exposed an important representation mismatch: the first
acceleration objective used latent dimensions 135--200, but official sampling
and evaluation reconstruct joints by FK from translation plus rotation
dimensions 0--134. A differentiable column-6D rotation conversion and
22-joint WoodenMesh global-rotation chain were therefore added and verified
against the cached raw IMU orientation (mean absolute error `7.93e-8`, maximum
`8.34e-7` over 16 records). A second experiment supervised active-sensor global
orientation on this exact FK path with weight `0.5`. Training completed at
`outputs/hymotion/prescreen512x2_stride4_early3_fkorient05_20260921` (adapter
SHA-256 `c204a0b73d132c3e794d60f77b38f5d8c3281bdc6e7518c07fd928afc3e8b124`).
The fixed-16 evaluation is
`outputs/hymotion/prescreen512x2_stride4_early3_fkorient05_eval16_20260921`.
Paired-vs-shuffled was 68.75% head and 81.25% wrists; paired-vs-zero was 62.50%
and 81.25%. Paired jerk was 90.19/87.99 m/s^3, or 2.39x/2.33x Text-only. This
also fails the discrimination/jerk gate and must not advance to 128 records or
full training.

These screens rule out simple one-step endpoint acceleration or orientation
auxiliary losses at the tested calibration. The original first-three-block
screen remains the best balanced HY-Motion configuration, but it is still not
eligible for full training. A subsequent experiment should change how control
is injected or regularized across the sampling trajectory, rather than scan
another scalar auxiliary-loss weight.

A third screen changed the sampling trajectory rather than adding an endpoint
loss. The new `control_timestep_power` adapter setting multiplies every control
residual by `(1 - t)^power`; its default `0` is exactly backward compatible.
At `power=1`, IMU control is strongest at the noise endpoint and vanishes at
the final data endpoint, preventing late Euler steps from adding new control
residuals. Direct tests verify exact Text-only equivalence at `t=1`.

The 1,024-step run completed at
`outputs/hymotion/prescreen512x2_stride4_early3_tgate1_20260921` (adapter
SHA-256 `b7df0de20ef0adfa61ab35865eeafbe5273f500d0454d65aea5fbcc35a6a7985`).
Its fixed-16 evaluation at
`outputs/hymotion/prescreen512x2_stride4_early3_tgate1_eval16_20260921` reached
paired-vs-shuffled fractions of 93.75% head and 81.25% wrists, the strongest
joint result so far. Paired-vs-zero was 68.75%/81.25%. Wrists paired jerk fell
to 62.00 m/s^3 (1.64x Text-only), but head paired jerk remained 82.31 m/s^3
(2.18x). This is the new best discrimination/smoothness trade-off, but it still
fails the mandatory relative-jerk gate and therefore must not advance to 128
records or full training. Do not scan adjacent timestep powers unless the
selection rule is explicitly changed; the next mechanism must directly bound
the temporal variation of the injected residual while preserving this early
trajectory control behavior.

A direct injected-residual curvature regularizer tested that hypothesis. It
penalizes the mean squared second temporal difference of the actual residuals
added to the three active DiT blocks. The expected gated curvature of the
trained `tgate1` adapter was `7.56e-6`; weight `1000` therefore targeted about
5% of the median second-epoch flow loss. The 1,024-step run completed at
`outputs/hymotion/prescreen512x2_stride4_early3_tgate1_curv1k_20260921`
(adapter SHA-256
`63820287432a2624d68ce0cd0126ddaf3f829fe4f03e6a842de7329b9c1c91b3`).
Its fixed-16 evaluation is
`outputs/hymotion/prescreen512x2_stride4_early3_tgate1_curv1k_eval16_20260921`.
Head/wrists paired-vs-shuffled fell to 87.50%/68.75%, and paired jerk changed
to 90.88/63.11 m/s^3 (2.41x/1.67x Text-only). This is worse than unregularized
`tgate1` on both selection criteria. It demonstrates that smoothness in the
1,024-dimensional injected feature residual is not a reliable proxy for
physical FK jerk; do not increase or scan this regularizer.

Inference-time hard cutoff was then tested on the unregularized `tgate1`
checkpoint without retraining. `control_timestep_cutoff` defaults to `1.0`; a
smaller value disables control completely at and after that flow timestep.
Unit tests verify that a disabled timestep is exactly equal to Text-only.
Cutoff `0.5` preserved paired-vs-shuffled fractions of 93.75%/81.25% and
reduced paired jerk to 46.35/43.76 m/s^3 (1.23x/1.16x Text-only). Its output is
`outputs/hymotion/prescreen512x2_stride4_early3_tgate1_cutoff05_eval16_20260921`.

One predeclared boundary follow-up at cutoff `0.35` is stored at
`outputs/hymotion/prescreen512x2_stride4_early3_tgate1_cutoff035_eval16_20260921`.
It again preserved paired-vs-shuffled at 93.75% head and 81.25% wrists;
paired-vs-zero was 75.00%/68.75%. Paired jerk fell to 42.22/41.96 m/s^3,
only 1.119x/1.112x Text-only. A deterministic 20,000-sample paired bootstrap
put paired-minus-Text-only mean jerk at 4.50 m/s^3 (95% CI -3.67 to 12.83,
sign-flip p=0.315) for head and 4.24 (-4.27 to 13.20, p=0.368) for wrists.
Paired-minus-shuffled IMU-score confidence intervals were strictly negative:
-0.721 (-1.046 to -0.414, p=0.00015) head and -0.643 (-1.138 to -0.219,
p=0.0112) wrists.

Thus cutoff `0.35` provides statistically supported instance-level control and
no statistically established jerk regression on 16 records. Nevertheless its
mean jerk ratios remain above the handoff's strict 1.0 no-regression rule, so
the automatic gate remains failed: do not start the 128-record validation or
7,009-record training without an explicit decision to replace the strict mean
gate with a statistical non-inferiority criterion. Stop cutoff scanning here;
adjacent thresholds would overfit the fixed-16 screen.

### Untouched 64-Record Check

The locked `tgate1`, cutoff-`0.35` configuration was evaluated on validation
indices 16--79, which had not been used in the preceding screens. The two GPU
shards and audited merge are:

- `outputs/hymotion/tgate1_cutoff035_unseen64_shard16_48_20260921`
- `outputs/hymotion/tgate1_cutoff035_unseen64_shard48_80_20260921`
- `outputs/hymotion/tgate1_cutoff035_unseen64_merged_20260921`

The merge contains 64 records and 512 generated outputs. It passed coverage,
train/validation disjointness, metric recomputation, rotation validity, and
Text-only identity between sensor configurations. Paired beat shuffled on
78.125% of head records and 70.3125% of wrist records; the paired-minus-shuffled
IMU-score bootstrap CIs were strictly negative for both. The effect therefore
generalizes beyond the original 16 records.

The jerk regression also generalized. Mean paired/Text-only jerk was 1.286x
head and 1.248x wrists. Paired-minus-Text-only mean jerk was +13.77 m/s^3
(95% CI 7.33 to 20.84, sign-flip p=0) and +11.95 m/s^3 (4.79 to 19.94,
p=0.0011), respectively. The increases were broad rather than caused only by
a few records: medians were +11.87/+8.02 m/s^3 and only 20.3%/21.9% of paired
outputs were no jerkier than Text-only. This rules out 7,009-record training
for the configuration as currently sampled.

Ground-truth context does not eliminate the concern. Across the same 64
records, reconstructed GT jerk averaged 69.90 m/s^3 versus 48.14 Text-only and
61.91/60.10 paired, so the strict Text-only gate also penalizes some genuine
dynamic motion. However, paired was closer than Text-only to each record's GT
jerk on only 51.6% head and 62.5% wrists, with no significant paired advantage
in bootstrap absolute-error differences. Root and articulated motion both
contributed: paired root-only jerk was 43.46/41.61 versus 32.85 Text-only, and
root-relative jerk was 41.39/40.79 versus 31.74. Filtering acceleration at the
same five-frame bandwidth used by the IMU metric left the paired/Text-only jerk
ratio near 1.29, so the finding is not merely an unfiltered endpoint artifact.

### Sampling-Space Diagnostics

Geodesic SO(3) output interpolation between independently generated Text-only
and paired motions was evaluated entirely on CPU. The corrected implementation
in `scripts/evaluate_hymotion_output_blending.py` reproduces both endpoints to
maximum absolute error `6.74e-6`; its valid report is
`outputs/hymotion/tgate1_cutoff035_unseen64_output_blend_v2_20260921`.
Interpolation is rejected: even alpha `0.05` raised paired/Text-only jerk to
1.240x head and 1.152x wrists, and intermediate alphas were worse. Do not blend
independently sampled final motions.

A more principled inference path instead computes conditioned and
unconditioned velocities at the same diffusion state, applies one binomial
three-point temporal filter to only their difference, and adds it back to the
unconditioned velocity. `control_delta_smoothing=0` is exactly backward
compatible. Eight direct control tests pass, including exact disabled behavior,
padding preservation, and the new difference-filter identity. The real-model
smoke peaked at 4.86 GB; the fixed-16 run peaked at 6.21 GB and took 1,061 s.

The fixed-16 artifact is
`outputs/hymotion/tgate1_cutoff035_delta_smooth1_eval16_20260921`. It preserves
paired-vs-shuffled discrimination at 93.75% head and 81.25% wrists. It improves
mean jerk over unfiltered cutoff `0.35`, from 1.119x/1.112x to 1.082x/1.089x,
but still fails the strict gate. Paired-minus-Text-only jerk was +3.10 m/s^3
(95% CI -4.82 to 11.07) head and +3.37 (-5.03 to 12.09) wrists. The paired
IMU-score advantage remained significant, with CIs -1.050 to -0.410 and
-1.114 to -0.216. This is a useful direction, not a passing result. Do not use
the remaining validation indices 80--127 or start full training unless a
predeclared stronger difference filter first passes the development screen.

The single predeclared stronger boundary used four applications of the same
filter; no intermediate pass counts were tested. Its artifact is
`outputs/hymotion/tgate1_cutoff035_delta_smooth4_eval16_20260921`. It again
preserved paired-vs-shuffled at 93.75%/81.25%, but jerk was still 1.050x head
and 1.072x wrists. The paired-minus-Text-only means were +1.88 m/s^3 (95% CI
-5.82 to 9.49) and +2.72 (-5.49 to 11.18). The significant IMU-score advantage
also remained. This is diminishing improvement without a strict pass, so stop
scanning filter strength or pass count.

An adapter-only CPU modality ablation over the unseen 64 records is stored in
`outputs/hymotion/tgate1_adapter_modality_analysis_unseen64_v2_20260921`; the
similarly named report without `v2` used the wrong tensor axis for temporal
differences and is invalid. Orientation dominates the learned residual
amplitude: orientation-vs-zero RMS was 0.139 head and 0.0830 wrists, versus
0.0108 and 0.0200 for acceleration. Acceleration has more curvature relative
to its much smaller amplitude, but none of residual amplitude, first
difference, or curvature strongly predicts observed output jerk (absolute
correlations at most 0.28). This agrees with the failed hidden-residual
curvature loss: the next mechanism should constrain the actual 201-dimensional
motion velocity, ideally using skeleton-aware output channels, rather than
regularize the 1,024-dimensional adapter representation again.

A first skeleton-aware output experiment was implemented as the opt-in
`control_delta_projection=sensor_chains`. It retains root translation and only
the local-rotation channels on each active sensor's WoodenMesh ancestor chain
(39 of 201 channels for head and 75 for both wrists). Nine direct tests verify
the channel sets and backward-compatible default. The one-record real-model
smoke is
`outputs/hymotion/tgate1_cutoff035_delta_smooth4_chains_smoke1_20260921`.
It failed decisively and therefore did not receive a 16-record run: head jerk
was 1.150x Text-only and paired lost to zero, while wrist paired IMU score was
worse than Text-only and jerk was 1.092x. Hard projection after the backbone
apparently breaks the coupled 201-dimensional flow state even though final
motion decoding explicitly uses translation and rotation channels. Retain the
option as a negative diagnostic; do not promote it. A future skeleton-aware
mechanism must be trained with the projection or use a representation-level
consistency objective, not impose the mask only at inference.

### Direct Output-Velocity Controller

The next controller moves the learnable IMU correction from the hidden DiT
features to the actual 201-dimensional flow velocity. Set
`control_mode=output` in `HYMotionIMUConfig`; the default remains `hidden`, so
old checkpoints and callers retain their previous behavior. The controller
reuses the sensor/temporal IMU encoder, conditions its zero-initialized output
head on the current noisy motion and flow timestep, and adds the resulting
velocity residual after the frozen official backbone. Its optional
`output_velocity_smoothing_passes` applies the same binomial three-point filter
to the residual during both training and inference. This makes the smoothness
constraint part of the trained mechanism instead of an inference-only
modification. `--max-steps` was also added for bounded integration smokes.

The two-step real-model smoke is
`outputs/hymotion/output_velocity_stride4_tgate1_smooth4_smoke2_20260921`;
its one-record controlled generation is in the corresponding `smoke2_eval1`
directory. Training peaked at 2.01 GB and generation at 4.79 GB. Direct tests
cover exact zero initialization, a frozen backbone, output-head gradients,
valid-frame masking, and reporting curvature from the actual filtered velocity
residual.

The first 512-record, two-epoch screen is
`outputs/hymotion/prescreen512x2_output_velocity_stride4_tgate1_smooth4_20260921`.
It completed 1,024 finite optimization steps. Mean flow loss fell from 0.612
to 0.527, with 0.473 over the last 128 steps. Its fixed-16 evaluation at
`outputs/hymotion/prescreen512x2_output_velocity_stride4_tgate1_smooth4_eval16_20260921`
nearly passed: paired-vs-shuffled was 68.75%/93.75% head/wrists and paired jerk
was 1.0089x/1.0014x Text-only.

One predeclared convergence extension, with no architecture or loss change,
trained the same 512 records for four epochs at
`outputs/hymotion/prescreen512x4_output_velocity_stride4_tgate1_smooth4_20260921`.
It completed 2,048 finite steps in 314.29 seconds, peaked at 2.02 GB, and has
adapter SHA-256
`802518836107aae12e4c9822351a62890ffead9fe44e4e35be26a192f9f72d12`.
Epoch mean flow losses were 0.612, 0.527, 0.474, and 0.380; the final 128-step
mean was 0.305. On the fixed-16 development set, paired beat shuffled on
75.0%/87.5% and zero on 100%/100% of head/wrist records. Mean paired/Text-only
jerk was 0.9979x/0.9984x. Paired-minus-shuffled IMU-score bootstrap CIs were
strictly negative for both sensors. This was the first configuration to pass
both fixed-development gates.

### Locked Output-Controller Confirmation

The four-epoch output controller was then evaluated without further tuning on
the previously untouched validation indices 80--127. The two generation shards
and audited merge are:

- `outputs/hymotion/output_velocity_x4_locked48_shard80_104_20260921`
- `outputs/hymotion/output_velocity_x4_locked48_shard104_128_20260921`
- `outputs/hymotion/output_velocity_x4_locked48_merged_20260921`

The merge contains 48 records and 384 generated outputs. It verifies complete
coverage, disjoint train/validation IDs, the exact adapter and backbone hashes,
real-backbone zero initialization, RoPE placement, recomputed metrics, valid
rotations, and identical Text-only outputs between sensor configurations. The
maximum rotation orthogonality error was `1.43e-6`.

Control discrimination generalized strongly. Paired beat shuffled on 39/48
(81.25%) head records and 46/48 (95.83%) wrist records; paired beat zero on
81.25% and 89.58%. Paired-minus-shuffled mean IMU score was -0.01261 head
(95% bootstrap CI -0.02238 to -0.00556, sign-flip p=0) and -0.10391 wrists
(-0.18367 to -0.05031, p=0). Thus the IMU input is measurably controlling the
generated trajectory rather than merely perturbing a Text-only sample.

The predeclared strict jerk result is mixed. Head paired jerk was 51.154 versus
51.262 m/s^3 Text-only, a passing ratio of 0.99789. Wrist paired jerk was
51.329 versus 51.262, a ratio of 1.00129; the +0.066 m/s^3 difference has a
95% bootstrap CI of -0.587 to +0.781 and sign-flip p=0.857. Although this is a
negligible and statistically unsupported increase, the locked rule required
the paired mean not to increase at all. The wrist result therefore fails the
literal gate. Do not retroactively relabel this run as a strict pass, and do
not start the 7,009-record training under the existing approval rule. The
scientifically defensible next decision is either to stop here or to declare a
practical non-inferiority margin in advance and test it on a fresh, untouched
validation subset; the current 48 records cannot be reused to choose that
margin.

### Fresh Non-Inferiority Confirmation Preregistration (2026-09-22)

Before preparing or generating any additional validation output, the next
confirmation is fixed as follows. It uses the unchanged four-epoch adapter
SHA-256 `802518836107aae12e4c9822351a62890ffead9fe44e4e35be26a192f9f72d12`,
unchanged sampling settings (50 Euler steps, text CFG 5, control strength 1,
timestep cutoff 1, and no inference-only smoothing), and deterministic
validation selection seed 1234. A 192-record cache must reproduce the existing
first 128 record IDs exactly; only the new indices 128--191 may enter this
test. Evaluation seeds remain `20260909 + global_index`, and the shuffled IMU
partner remains the next record cyclically within the full 192-record list.

The new quantitative gate must pass for both head and wrists:

1. Paired must beat shuffled on at least 70% of records.
2. The upper end of the two-sided 95% paired bootstrap CI for paired-minus-
   shuffled IMU score must be below zero.
3. The upper end of the two-sided 95% paired bootstrap CI for paired-minus-
   Text-only mean jerk must be no greater than 2% of that sensor's Text-only
   mean jerk.

The third rule is a conservative practical non-inferiority test: it uses the
97.5th bootstrap percentile even though a conventional one-sided 95% test
would use the 95th percentile. The original strict zero-increase `jerk_gate`
will still be reported separately and is not retroactively changed. No model,
margin, index, or inference setting may be adjusted after observing indices
128--191. Passing this quantitative gate still requires a semantic visual
review before full training is authorized; failure stops this controller from
advancing to the 7,009-record run.

The preparation script supports `--extend-prepared-data` for this test. It
requires `--prepare-only`, verifies that the old record list is an exact prefix
selected with the same backbone checkpoint, symlinks the old per-record NPZ/PT
files, and computes only the newly selected suffix. The intended command is:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  outputs/envs/hymotion/bin/python scripts/train_hymotion_imu.py \
  --hymotion-root /home/4T-2/group_motion/relatedworks/HY-Motion-1.0 \
  --runtime-root outputs/hymotion/runtime_root \
  --manifest outputs/manifests_full/val_standard_imu.jsonl \
  --texts /home/4T-2/group_motion/relatedworks/HumanML3D/HumanML3D/texts \
  --output outputs/hymotion/prescreen_val192_20260922 \
  --device cuda:0 --count 192 --epochs 1 --seed 1234 --prepare-only \
  --extend-prepared-data outputs/hymotion/prescreen_val128_20260909
```

Preparation now keeps the unused motion transformer on CPU and moves only the
exact unquantized Qwen3-8B/CLIP text encoder to the target GPU. This lowers the
guard for `--prepare-only` from 23 GiB to 18 GiB without changing text weights,
precision, or embeddings. Ordinary training still requires 23 GiB when data
are not already prepared. Do not change or skip the text encoder to fit a more
crowded GPU; that would make new records incomparable with the existing
validation protocol.

### Output-Controller Exploratory Replication (2026-09-22)

While cuda:0 had enough space for generation but not the 23 GiB text-cache
preparation, the unchanged output controller was evaluated on indices 16--79.
These records had previously informed hidden-controller diagnostics, so this is
an exploratory replication rather than the fresh preregistered confirmation.
The generation and audited artifacts are:

- `outputs/hymotion/output_velocity_x4_exploratory64_16_80_20260922`
- `outputs/hymotion/output_velocity_x4_exploratory64_16_80_audited_20260922`

All 64 records and 512 outputs completed in 3,206.25 seconds with a 6.14 GB
PyTorch peak. The audit passed complete coverage, train/validation disjointness,
metric recomputation, rotation validity (maximum orthogonality error `1.55e-6`),
checkpoint hashes, zero initialization, RoPE placement, and Text-only identity.

The result independently supports the mechanism. Paired beat shuffled on
78.125% head and 82.8125% wrist records. Paired-minus-shuffled IMU-score CIs
were -0.01570 to -0.00469 head and -0.05196 to -0.00037 wrists. Mean paired
jerk was lower than Text-only for both: 47.677 versus 48.142 m/s^3 head
(0.9903x), and 47.922 versus 48.142 wrists (0.9954x). The paired-minus-Text-only
jerk CI upper bounds were +0.145 and +0.358 m/s^3, both below the preregistered
2% margin of 0.963 m/s^3. Thus this exploratory set passes both the original
strict mean gate and the new non-inferiority gate, but it cannot authorize full
training because it is not the untouched 128--191 set.

At the end of this run cuda:0 remained occupied by three unrelated processes:
30,384 MiB used, 15,116 MiB free, and 100% utilization. The 192-record cache
preparation therefore remains pending. CPU selection verification already
proved that `count=192, seed=1234` reproduces all first 128 records exactly and
adds 64 unique records (first `012712`, last `004308`).

### Fresh 128--191 Confirmation Result (2026-09-23)

The deterministic cache extension completed on cuda:3 at
`outputs/hymotion/prescreen_val192_20260922`. It contains 192 records: the first
128 NPZ/PT pairs are links to the previously audited cache, and only the new 64
records were reconstructed and text-encoded. Keeping the unused motion
transformer on CPU allowed the exact unquantized Qwen3-8B/CLIP encoder to fit
with about 19.3 GiB initially free. The full cache audit is
`outputs/a40_audit/hymotion_val192_control_data_20260923.json`: all text tensors
are finite and nonzero, maximum IMU round-trip error is `0.0005149841`, and
maximum root-relative position error is `9.54e-7` m.

The preregistered indices 128--191 then completed on cuda:3 without restart:

- generation: `outputs/hymotion/output_velocity_x4_fresh64_128_192_20260923`
- audited statistics: `outputs/hymotion/output_velocity_x4_fresh64_128_192_audited_20260923`

The run generated all 512 outputs in 3,969.00 seconds and peaked at 6.14 GB of
PyTorch allocation. The audit passed coverage, train/validation disjointness,
checkpoint hashes, zero initialization, RoPE placement, metric recomputation,
rotation validity (maximum orthogonality error `1.43e-6`), and Text-only
identity between sensor configurations.

IMU control passes decisively. Paired beat shuffled on 50/64 head records
(78.125%) and 59/64 wrist records (92.1875%). Paired-minus-shuffled IMU-score
bootstrap CIs were -0.01465 to -0.00496 head and -0.10166 to -0.02700 wrists;
both the win-rate and CI gates pass.

Mean jerk also improved: paired/Text-only was 49.192/49.726 m/s^3 head
(0.98926x) and 49.232/49.726 wrists (0.99006x). Both therefore pass the old
strict mean non-increase gate. However, the deliberately conservative
preregistered non-inferiority rule used the 97.5th bootstrap percentile. The
2% absolute margin was 0.99453 m/s^3. The head paired-minus-Text-only jerk CI
was -2.2094 to +1.0421, exceeding the margin by 0.0476 m/s^3; wrists was
-2.0949 to +0.9464 and passed. Therefore the joint preregistered gate fails on
head despite the better point estimate and very small boundary miss.

Do not change the bootstrap seed, substitute a conventional one-sided 95%
bound, or relax the 2% margin after seeing this result. Under the preregistered
decision rule this controller is not authorized for 7,009-record training.
The evidence still establishes reproducible IMU control with no observed mean
jerk regression across the locked and fresh sets; future work should either
stop here or define a new model/experiment with a genuinely new test set, not
reuse indices 128--191 to tune another threshold.

Because cuda:3 became crowded during generation, `evaluate_hymotion_imu.py`
now has an opt-in strict `--resume` path. It validates every existing key,
index, seed, condition setting, shuffled partner, and NPZ metadata before
skipping completed outputs. The formal run did not need it and completed in a
single process.
