import numpy as np
import pytest

from itm.data.humanml_clock import notebook_source_indices


def test_near_sixty_fps_replays_two_stride_without_rounding():
    plan = notebook_source_indices(source_path="SSM_synced/example.npz", source_fps=59.9999885559082,
                                   source_frames=315, start_frame=0, end_frame=158)
    assert plan["integer_stride"] == 2
    assert plan["predicted_feature_frames"] == 157
    assert plan["raw_source_indices"][-1] == 314
    assert plan["nominal_to_physical_duration_ratio"] == pytest.approx(1.5, abs=1e-6)
    assert not plan["permitted_as_deployable_input_transform"]


def test_exact_sixty_and_dataset_trim_use_distinct_source_origin():
    plan = notebook_source_indices(source_path="AMASS/MPI_HDM05/a.npz", source_fps=60.,
                                   source_frames=600, start_frame=10, end_frame=30)
    assert plan["integer_stride"] == 3
    np.testing.assert_array_equal(plan["raw_source_indices"], np.arange(70, 90) * 3)
    assert plan["source_times_seconds"][0] == 3.5
    assert plan["nominal_clip_times_seconds"][0] == 0.
    assert plan["nominal_to_physical_duration_ratio"] == 1.


def test_notebook_clamping_is_explicit_diagnostic():
    plan = notebook_source_indices(source_path="SSM_synced/a.npz", source_fps=60.,
                                   source_frames=30, start_frame=8, end_frame=15)
    assert plan["notebook_end_would_clamp"]
    assert plan["predicted_feature_frames"] == 1


def test_invalid_stride_is_rejected():
    with pytest.raises(ValueError):
        notebook_source_indices(source_path="a", source_fps=19., source_frames=30, start_frame=0, end_frame=10)
