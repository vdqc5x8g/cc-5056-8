import numpy as np
import pytest

from wraplight.band import band_from_alpha, band_from_mask
from wraplight.colorspace import linear_to_srgb, srgb_to_linear
from wraplight.composite import composite_base
from wraplight.deformer import apply_deformer, strand_gate
from wraplight.params import WrapParams
from wraplight.pipeline import run_pipeline

SIZE = 96


def make_subject(size=SIZE):
    """竖条前景, 硬边."""
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


def run(params, **kw):
    fg, alpha = make_subject()
    bg = kw.pop("bg", make_bg_with_source())
    return run_pipeline(fg, alpha, bg, params, **kw)


@pytest.mark.parametrize("method", ["blur", "falloff", "visibility"])
def test_uniform_background_gives_zero_wrap(method):
    """均匀背景下周边不超出局部背景, 包光必须为零(不重复计入背景)."""
    fg, alpha = make_subject()
    bg = np.full((SIZE, SIZE, 3), 0.4)
    params = WrapParams(method=method)
    out = run_pipeline(fg, alpha, bg, params)
    assert np.abs(out.wrap_layer).max() < 1e-9


@pytest.mark.parametrize("method", ["blur", "falloff", "visibility"])
def test_interior_and_far_background_preserved(method):
    """内部主体与远处背景不被包光改变."""
    params = WrapParams(method=method)
    out = run(params)
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    fg_lin = srgb_to_linear(fg)
    bg_lin = srgb_to_linear(np.clip(bg, 0, None))
    inside = out.signed_dist > params.inner_width + 1.0
    assert inside.any()
    np.testing.assert_allclose(
        out.composite_on[inside], fg_lin[inside], atol=1e-9)
    far = out.signed_dist < -(params.outer_width + 1.0)
    assert far.any()
    np.testing.assert_allclose(
        out.composite_on[far], bg_lin[far], atol=1e-9)


def test_occluder_stops_wrap_for_visibility_not_blur():
    """亮物移到遮挡面后: visibility 法停止包光, blur 法不受影响."""
    occ = make_occluder()
    p_vis = WrapParams(method="visibility")
    p_blur = WrapParams(method="blur")
    out_vis_open = run(p_vis, occluder=None)
    out_vis_occ = run(p_vis, occluder=occ)
    out_blur_open = run(p_blur, occluder=None)
    out_blur_occ = run(p_blur, occluder=occ)

    energy = lambda o: np.linalg.norm(o.wrap_layer, axis=-1)
    roi = np.zeros((SIZE, SIZE), bool)
    roi[40:56, 42:48] = True   # 主体左侧边缘, 挡板正后方
    roi &= out_vis_open.band_weight > 1e-3
    assert roi.any()

    e_open = energy(out_vis_open)[roi].mean()
    e_occ = energy(out_vis_occ)[roi].mean()
    assert e_open > 1e-6
    assert e_occ < 0.35 * e_open

    b_open = energy(out_blur_open)[roi].mean()
    b_occ = energy(out_blur_occ)[roi].mean()
    assert b_open > 1e-6
    assert abs(b_occ - b_open) < 1e-9


def test_directionality_of_wrap():
    """亮源只在一侧: 该侧边缘贡献应显著大于对侧."""
    params = WrapParams(method="visibility")
    out = run(params)
    energy = np.linalg.norm(out.wrap_layer, axis=-1)
    band = out.band_weight > 1e-3
    left = np.zeros((SIZE, SIZE), bool)
    left[:, 42:48] = True
    right = np.zeros((SIZE, SIZE), bool)
    right[:, 64:70] = True
    e_left = energy[band & left].mean()
    e_right = energy[band & right].mean()
    assert e_left > 3.0 * e_right


def test_wrap_does_not_double_count_background():
    """线性光度下包光只加超出部分: 合成结果不超过 基底色+超出量."""
    params = WrapParams(method="visibility", intensity=1.0)
    out = run(params)
    base = composite_base(
        srgb_to_linear(make_subject()[0]), make_subject()[1],
        srgb_to_linear(make_bg_with_source()))
    np.testing.assert_allclose(
        out.composite_on, base + out.wrap_layer, atol=1e-12)
    band = out.band_weight > 1e-3
    excess_norm = np.linalg.norm(out.excess, axis=-1)
    wrap_norm = np.linalg.norm(out.wrap_layer, axis=-1)
    assert (wrap_norm[band] <= excess_norm[band] + 1e-9).all()


def test_band_scales_with_width_and_preserves_interior():
    """不同尺度: 边缘带随宽度变化, 内部始终保持不变."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    narrow = run_pipeline(fg, alpha, bg, WrapParams(inner_width=2, outer_width=4))
    wide = run_pipeline(fg, alpha, bg, WrapParams(inner_width=10, outer_width=20))
    n_narrow = (narrow.band_weight > 1e-3).sum()
    n_wide = (wide.band_weight > 1e-3).sum()
    assert n_wide > 2 * n_narrow
    fg_lin = srgb_to_linear(fg)
    for out, p in ((narrow, WrapParams(inner_width=2, outer_width=4)),
                   (wide, WrapParams(inner_width=10, outer_width=20))):
        inside = out.signed_dist > p.inner_width + 1.0
        np.testing.assert_allclose(
            out.composite_on[inside], fg_lin[inside], atol=1e-9)


def test_band_from_explicit_geometry():
    """边缘范围也可来自明确的几何 mask."""
    _, alpha = make_subject()
    mask = np.zeros_like(alpha)
    mask[30:60, 44:50] = 1.0
    band = band_from_mask(mask, alpha)
    assert (band.weight > 0).sum() == int(mask.sum())
    assert band.weight[40, 46] == 1.0


def test_deformer_relocates_and_gates():
    """前景变形器: 位移场改变层分布, 发丝门控限制在细碎结构上."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    base = WrapParams(method="visibility")
    out_plain = run_pipeline(fg, alpha, bg, base)
    deformed = WrapParams(method="visibility", deform_amp=3.0, strand_strength=0.8)
    out_def = run_pipeline(fg, alpha, bg, deformed)
    assert not np.allclose(out_plain.wrap_layer, out_def.wrap_layer)
    gate = strand_gate(alpha)
    assert 0.0 <= gate.min() and gate.max() <= 1.0


def test_params_json_roundtrip(tmp_path):
    p = WrapParams(method="blur", blur_sigma=7.5, use_occluder=False)
    path = tmp_path / "p.json"
    p.save(str(path))
    loaded = WrapParams.load(str(path))
    assert loaded.method == "blur"
    assert loaded.blur_sigma == 7.5
    assert loaded.use_occluder is False


def test_srgb_roundtrip():
    img = np.random.default_rng(0).random((16, 16, 3))
    np.testing.assert_allclose(
        linear_to_srgb(srgb_to_linear(img)), img, atol=1e-9)


def test_linear_mode_off_is_flagged_difference():
    """linear=off 的对照模式应产生与线性模式不同的(更亮的)结果."""
    fg, alpha = make_subject()
    bg = make_bg_with_source()
    on = run_pipeline(fg, alpha, bg, WrapParams(linear=True))
    off = run_pipeline(fg, alpha, bg, WrapParams(linear=False))
    band = on.band_weight > 1e-3
    assert not np.allclose(on.composite_on[band], off.composite_on[band])
