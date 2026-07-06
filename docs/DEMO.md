# ITM Condition Lab

The demo uses the test split by default and randomly selects samples A and B.
Selections remain editable before generation.

## Views

- Four-way: Ground truth A, Text A only, IMU A only, and Text A + IMU A.
- Matrix: Ground truth A/B plus all eight non-empty Text/IMU combinations.

All generated panels in one run share the same initial diffusion noise. Motion
panels play synchronously and can be rotated independently. Target IMU A/B
signals are shown below the motion grid.

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

Only one generation runs at a time. A concurrent request receives HTTP 409;
the service never terminates or preempts another GPU process.
