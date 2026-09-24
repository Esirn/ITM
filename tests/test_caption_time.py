import numpy as np
import pytest

from itm.data.caption_time import caption_interval, parse_timed_caption, plan_caption_crop


@pytest.mark.parametrize("line", ["hello", "hello##bad#0", "hello##nan#0", "hello##0#inf",
                                 "hello##-1#2", "hello##2#1", "##0#0"])
def test_bad_tags_never_turn_into_full_clip(line):
    with pytest.raises(ValueError):
        parse_timed_caption(line)


def test_full_clip_sentinel_and_model_limit():
    caption = parse_timed_caption("walk##0#0")
    assert caption_interval(caption, 199)["frame_slice"] == [0, 199]
    with pytest.raises(ValueError, match="model limit"):
        plan_caption_crop(caption, 199, np.arange(300) / 30)


def test_offset_crop_uses_original_imu_clock_not_rebased_clock():
    plan = plan_caption_crop(parse_timed_caption("wave##1#2"), 60, np.arange(91) / 30)
    np.testing.assert_allclose(plan["source_times_seconds"], np.arange(20, 40) / 20)
    np.testing.assert_allclose(plan["model_times_seconds"], np.arange(20) / 20)
    assert plan["frame_slice"] == [20, 40]
    assert plan["source_sample_interval_seconds"] == [1., 1.95]
    assert not plan["training_admissible"]


def test_tagged_end_is_exclusive_and_quantization_is_recorded():
    interval = caption_interval(parse_timed_caption("wave##0.02#1.02"), 30)
    assert interval["frame_slice"] == [0, 20]
    np.testing.assert_allclose(interval["tag_quantization_seconds"], [-.02, -.02])
    with pytest.raises(ValueError, match="outside feature support"):
        caption_interval(parse_timed_caption("wave##0#1.50001"), 30)


def test_short_imu_cannot_silently_shorten_caption():
    with pytest.raises(ValueError, match="outside source support"):
        plan_caption_crop(parse_timed_caption("wave##1#2"), 60, np.arange(30) / 30)
