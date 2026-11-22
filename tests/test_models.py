"""组合贡献模型与指标的测试."""

import numpy as np
import pytest

from wraplight import models
from wraplight.colorspace import srgb_to_linear
from wraplight.metrics import edge_blend_diff, edge_energy, subject_color_diff
from wraplight.params import WrapParams
from wraplight.pipeline import run_pipeline

SIZE = 96


def make_subject(size=SIZE):
    alpha = np.zeros((size, size))
    alpha[16:80, 48:64] = 1.0
    fg = np.zeros((size, size, 3))
    fg[:] = (0.2, 0.1, 0.05)
    return fg, alpha


def make_bg_with_source(size=SIZE, center=(48, 14), radius=7, value=1.0):
    bg = np.full((size, size, 3), 0.08)
    yy, xx = np.mgrid[0:size, 0:size]
    d2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    bg[d2 <= radius * radius] = value
    return bg


def make_occluder(size=SIZE, box=(40, 30, 56, 42)):
    occ = np.zeros((size, size))
    y0, x0, y1, x1 = box
    occ[y0:y1, x0:x1] = 1.0
    return occ


def combined_params(method, **kw):
    base = dict(method=method, depth_threshold=24.0, occ_threshold=0.25)
    base.update(kw)
    return WrapParams(**base)


# ---------- 门槛无默认值: 未显式设置即报错 ----------

@pytest.mark.parametrize("method,missing", [
    ("depth_select", "depth_threshold"),
    ("switch", "occ_threshold"),
])
def test_combined_requires_explicit_threshold(method, missing):
    """组合模型门槛默认 None, 用到而未显式设置必须报错(不以默认值遮错)."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    params = WrapParams(method=method)   # 门槛保持 None
    with pytest.raises(ValueError, match=missing):
        run_pipeline(fg, alpha, bg, params)


def test_require_threshold_message():
    p = WrapParams(method="depth_select")
    with pytest.raises(ValueError):
        models.require_threshold(p, "depth_threshold")
    p = WrapParams(method="depth_select", depth_threshold=16.0)
    assert models.require_threshold(p, "depth_threshold") == 16.0


# ---------- 组合逻辑(单元) ----------

def test_combine_depth_select_picks_per_pixel():
    excess_dir = np.full((4, 4, 3), 1.0)
    excess_local = np.full((4, 4, 3), 0.2)
    depth = np.array([[1.0, 1.0, 50.0, 50.0]] * 4)
    out, near = models.combine_depth_select(
        excess_dir, excess_local, depth, depth_threshold=24.0)
    assert near[:, :2].all() and not near[:, 2:].any()
    np.testing.assert_allclose(out[:, :2], 1.0)
    np.testing.assert_allclose(out[:, 2:], 0.2)


def test_combine_vis_mix_endpoints():
    """可见份额 1 -> 纯方向加权; 0 -> 纯局部平均."""
    excess_dir = np.full((2, 2, 3), 1.0)
    excess_local = np.full((2, 2, 3), 0.0)
    vis_one = np.ones((2, 2))
    vis_zero = np.zeros((2, 2))
    np.testing.assert_allclose(
        models.combine_vis_mix(excess_dir, excess_local, vis_one), 1.0)
    np.testing.assert_allclose(
        models.combine_vis_mix(excess_dir, excess_local, vis_zero), 0.0)
    half = models.combine_vis_mix(excess_dir, excess_local, np.full((2, 2), 0.5))
    np.testing.assert_allclose(half, 0.5)


def test_combine_switch_threshold():
    excess_dir = np.full((2, 2, 3), 1.0)
    excess_local = np.zeros((2, 2, 3))
    blocked = np.array([[0.0, 0.3], [0.1, 0.9]])
    out, use_dir = models.combine_switch(
        excess_dir, excess_local, blocked, occ_threshold=0.25)
    assert use_dir[0, 1] and use_dir[1, 1]
    assert not use_dir[0, 0] and not use_dir[1, 0]
    np.testing.assert_allclose(out[0, 0], 0.0)
    np.testing.assert_allclose(out[1, 1], 1.0)


# ---------- 组合模型(集成) ----------

@pytest.mark.parametrize("method", models.COMBINED_METHODS)
def test_combined_uniform_background_zero_wrap(method):
    """均匀背景下组合模型包光也必须为零."""
    fg, alpha = make_subject()
    bg = np.full((SIZE, SIZE, 3), 0.4)
    out = run_pipeline(fg, alpha, bg, combined_params(method))
    assert np.abs(out.wrap_layer).max() < 1e-9


@pytest.mark.parametrize("method", models.COMBINED_METHODS)
def test_combined_preserves_interior(method):
    """组合模型同样不改变主体内部(内部颜色保留)."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    params = combined_params(method)
    out = run_pipeline(fg, alpha, bg, params)
    fg_lin = srgb_to_linear(fg)
    inside = out.signed_dist > params.inner_width + 1.0
    np.testing.assert_allclose(
        out.composite_on[inside], fg_lin[inside], atol=1e-9)


def test_depth_select_matches_dir_near_source():
    """亮源附近(局部深度小)depth_select 应与方向加权一致."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    out = run_pipeline(fg, alpha, bg, combined_params("depth_select"))
    ref = run_pipeline(fg, alpha, bg, WrapParams(method="visibility"))
    near = out.debug["select_dir_mask"] & (out.band_weight > 1e-3)
    assert near.any()
    np.testing.assert_allclose(
        out.wrap_layer[near], ref.wrap_layer[near], atol=1e-9)


def test_switch_records_decision():
    """切换决定随 debug 保存(决定和采样证据随工程保存)."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    occ = make_occluder()
    out = run_pipeline(fg, alpha, bg, combined_params("switch"), occluder=occ)
    assert "decision" in out.debug
    assert "select_dir_mask" in out.debug


def test_occlusion_change_eliminates_blur():
    """背景亮度不变而遮挡改变: 局部平均不变(错误模型), 方向加权跟随."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    occ = make_occluder()
    energy = lambda o: np.linalg.norm(o.wrap_layer, axis=-1)

    out_blur_open = run_pipeline(fg, alpha, bg, WrapParams(method="blur"))
    out_blur_occ = run_pipeline(fg, alpha, bg, WrapParams(method="blur"),
                                occluder=occ)
    np.testing.assert_allclose(
        energy(out_blur_open), energy(out_blur_occ), atol=1e-12)

    out_sw_open = run_pipeline(fg, alpha, bg, combined_params("switch"))
    out_sw_occ = run_pipeline(fg, alpha, bg, combined_params("switch"),
                              occluder=occ)
    assert not np.allclose(energy(out_sw_open), energy(out_sw_occ))


# ---------- 指标复算 ----------

def test_metrics_subject_color_diff_zero_for_wrap():
    """包光不动内部: 主体色差应≈0; 边缘融合应>0."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    params = WrapParams(method="visibility")
    out = run_pipeline(fg, alpha, bg, params)
    fg_lin = srgb_to_linear(fg)
    scd = subject_color_diff(out.composite_on, fg_lin, out.signed_dist,
                             params.inner_width)
    assert scd["mean"] < 1e-9
    eb = edge_blend_diff(out.composite_on, out.composite_off, out.band_weight)
    assert eb["mean"] > 0.0
    ee = edge_energy(out.wrap_layer, out.band_weight)
    assert ee["mean"] > 0.0 and ee["band_pixels"] > 0


def test_metrics_global_blur_raises_subject_diff():
    """整体模糊对照: 主体色差显著大于包光模型(不能替代包光)."""
    from wraplight.blur import gaussian_blur
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    params = WrapParams(method="visibility")
    out = run_pipeline(fg, alpha, bg, params)
    fg_lin = srgb_to_linear(fg)
    blurred = gaussian_blur(out.composite_off, 6.0)
    scd_blur = subject_color_diff(blurred, fg_lin, out.signed_dist,
                                  params.inner_width)
    scd_wrap = subject_color_diff(out.composite_on, fg_lin, out.signed_dist,
                                  params.inner_width)
    assert scd_blur["mean"] > 1e-3
    assert scd_blur["mean"] > 100 * max(scd_wrap["mean"], 1e-12)


def test_pipeline_output_carries_metrics():
    """边缘能量及主体色差随流水线输出(对应执行证据)."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    out = run_pipeline(fg, alpha, bg, combined_params("vis_mix"))
    assert set(out.metrics) == {"edge_energy", "subject_color_diff", "edge_blend"}
    assert out.metrics["edge_energy"]["band_pixels"] > 0
