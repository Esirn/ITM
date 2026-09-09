# ITM Condition Lab

The demo uses the test split by default and randomly selects samples A and B.
Selections remain editable before generation.

## Views

Color settings is collapsed by default in the shared sidebar. Its swatches
control GT/generated skeletons and SMPL surfaces separately, motion and IMU
canvas colors, the three acceleration traces, UI surfaces, text, controls and
tags. Changes repaint paused previews immediately without refitting SMPL.
Preferences are stored in this browser; Restore default colors resets them.
SMPL colors are material colors and are shaded by the scene lighting.

- Four-way: Ground truth A, Text A only, IMU A only, and Text A + IMU A.
- Matrix: Ground truth A/B plus all eight non-empty Text/IMU combinations.

All generated panels in one run share the same initial diffusion noise. Motion
panels play synchronously and can be rotated independently. Target IMU A/B
signals are shown below the motion grid.

### SMPL preview

The shared sidebar Display selector offers Skeleton, SMPL, and SMPL + skeleton.
Selecting an SMPL mode fits up to two loaded/generated panels concurrently
(200 optimization iterations per panel). Skeletons remain visible
until each mesh is ready. A failed or busy fit can be retried with Retry SMPL.
The default remains Skeleton, so simply loading a result does not start fitting.

Fit device offers Auto, GPU All, CPU, GPU 0 and GPU 1. GPU All discovers all GPUs
and submits up to one parallel fit per GPU, waiting for busy GPUs without CPU
fallback. Cached results remain reusable regardless of the selected device.
Auto checks GPU load before each
fit (at least 6 GiB free and utilization below 20%), reserving at most one task
per device. It falls back to a single four-thread CPU worker when GPUs are busy.
Busy explicit devices are retried for up to ten minutes. Workers reuse the SMPL
model across fits. No new GPU fits start during demo generation; generation on
a GPU already fitting returns a busy message rather than interrupting that fit.
Load's Clear current SMPL cache removes all mesh caches corresponding to the
currently displayed result and returns to Skeleton mode. Original motions are
preserved. Identical motions share caches, so clearing also invalidates their
disk cache in other runs; active fits block clearing. Other open tabs may still
display their in-memory meshes until reloaded.

A 65-frame, 200-iteration check measured CPU fitting at 15.68 s and RTX 4090
fitting at 3.97 s (about 4x). Both produced mean joint error 1.76 cm. These are
fit-loop times, excluding Python startup, model loading and HTTP transfer;
timings and peak allocated GPU memory are recorded per cache. Benchmark files
are in `outputs/benchmarks/smpl_fit/`.

The preview uses the local neutral SMPL asset under the configured MDM root,
with fixed zero shape coefficients and an optimized sequence scale. All panels,
including Ground truth, are fitted from their displayed 22 HumanML joints;
these are not original SMPL ground-truth surfaces. The SMPL fit tag reports mean
joint fitting error in centimeters. Surface twist and hands remain ambiguous,
and fitting does not correct the original generation's semantics or foot sliding.

Results are cached by motion content and model identity in
`outputs/demo_meshes/<hash>/` (vertices.bin, faces.json, metadata.json, fit.log).
Meshes use the same frame index, root-centered camera, rotation controls and
layout as skeletons. Three.js 0.160.1 and its MIT license are vendored locally;
the browser does not fetch a CDN or download the SMPL model parameters.
SMPL display requires WebGL in the viewing browser; when it is unavailable,
the preview reports an error and keeps the skeleton visible.

Use the Size slider in the playback toolbar to resize motion panels. Holding
Ctrl while scrolling over the result area changes the same panel-size setting
without zooming the sidebar or text. Matrix mode keeps a fixed three-column
condition layout and scrolls horizontally when necessary.

## Storage

Each run is written to `outputs/demo_runs/<run_id>/` with the request, generated
condition specification, compressed NumPy output, browser JSON, and generation
log. Enter a run ID in the sidebar to reopen its browser JSON without using a
GPU.

The sidebar separates Generate and Load modes. Load lists recent completed runs
and also accepts an explicit run ID; switching modes does not clear the result
currently playing.

For experiment results, Load mode has a `Result set` selector. It maps to:

- `stage1`: `outputs/mdm_control/experiments/`
- `stage2`: `outputs/mdm_control/experiments_stage2/`
- `stage2b`: `outputs/mdm_control/experiments_stage2b_balanced_b/`
- `stage3`: `outputs/mdm_control/experiments_stage3_upper_body/`
- `stage4`: `outputs/mdm_control/experiments_stage4_text_anchor/`

Enter the path relative to the selected result set, for example:

```text
same_text_different_imu/test_walk_head_wrists
matrix_test_wrists/pair_001_004488_004222
same_head_imu_different_text/test_prompts
```

Loaded experiment pages display the selected root and relative path in the
result summary, so Stage-1/Stage-2b examples with identical experiment names
are not confused.

Only one generation runs at a time. A concurrent request receives HTTP 409;
the service never terminates or preempts another GPU process.
