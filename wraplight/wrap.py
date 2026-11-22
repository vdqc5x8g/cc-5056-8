from dataclasses import dataclass, field

import numpy as np

from .blur import gaussian_blur
from .rays import integrate_visible_background


@dataclass
class WrapResult:
    """包光估算输出.

    layer:  HxWx3 最终叠加到合成上的贡献(已乘边缘带权重与强度)
    excess: HxWx3 周边辐射超出局部背景的部分(未乘带权重), 调试用
    weight: HxW   实际作用权重(带权重 x 强度)
    debug:  各估算器的附加图层(可见性/遮挡比例等)
    """

    layer: np.ndarray
    excess: np.ndarray
    weight: np.ndarray
    debug: dict = field(default_factory=dict)


def _bilinear(img, ys, xs):
    h, w = img.shape[:2]
    ys = np.clip(ys, 0.0, h - 1.0)
    xs = np.clip(xs, 0.0, w - 1.0)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (ys - y0)[:, None] if img.ndim == 3 else (ys - y0)
    fx = (xs - x0)[:, None] if img.ndim == 3 else (xs - x0)
    return (
        img[y0, x0] * (1 - fy) * (1 - fx)
        + img[y0, x1] * (1 - fy) * fx
        + img[y1, x0] * fy * (1 - fx)
        + img[y1, x1] * fy * fx
    )


def _clip_excess(excess, allow_negative):
    if allow_negative:
        return excess
    return np.clip(excess, 0.0, None)


def wrap_blur(bg, alpha, band, params):
    """办法一: 从附近像素近似包光.

    对背景做高斯模糊, 并在朝背景侧偏移 dir_offset 处采样(方向限制),
    只取超出局部背景的部分, 作用范围由边缘带限定(距离限制) ——
    不是把整个背景模糊后叠到所有像素上.
    """
    blurred = gaussian_blur(bg, params.blur_sigma)
    ys, xs = np.nonzero(band.weight > 1e-4)
    excess = np.zeros_like(bg)
    if ys.size:
        normals = band.normal[ys, xs]
        sy = ys + normals[:, 0] * params.dir_offset
        sx = xs + normals[:, 1] * params.dir_offset
        sampled = _bilinear(blurred, sy, sx)
        local = bg[ys, xs]
        excess[ys, xs] = _clip_excess(sampled - local, params.allow_negative)
    weight = band.weight * params.intensity
    layer = excess * weight[:, :, None]
    return WrapResult(
        layer=layer,
        excess=excess,
        weight=weight,
        debug={"blurred_bg": blurred},
    )


def _wrap_via_rays(bg, alpha, band, params, occluder, use_occlusion):
    vis = integrate_visible_background(
        bg,
        alpha,
        band,
        occluder=occluder,
        ray_count=params.ray_count,
        max_dist=params.max_dist,
        cone_angle=params.cone_angle,
        falloff_tau=params.falloff_tau,
        use_occlusion=use_occlusion,
    )
    excess = _clip_excess(vis.integral - bg, params.allow_negative)
    weight = band.weight * params.intensity
    layer = excess * weight[:, :, None]
    return WrapResult(
        layer=layer,
        excess=excess,
        weight=weight,
        debug={
            "visibility": vis.visibility,
            "blocked": vis.blocked,
            "mean_hit_dist": vis.mean_hit_dist,
            "integral": vis.integral,
        },
    )


def wrap_falloff(bg, alpha, band, params, occluder=None):
    """办法二(对照): 仅距离衰减估算, 不查询遮挡."""
    return _wrap_via_rays(bg, alpha, band, params, occluder, use_occlusion=False)


def wrap_visibility(bg, alpha, band, params, occluder=None):
    """办法二: 根据背景可见区域积分, 遮挡物截停射线."""
    return _wrap_via_rays(bg, alpha, band, params, occluder, use_occlusion=True)


def compute_wrap(bg, alpha, band, params, occluder=None):
    occ = occluder if params.use_occluder else None
    if params.method == "blur":
        return wrap_blur(bg, alpha, band, params)
    if params.method == "falloff":
        return wrap_falloff(bg, alpha, band, params, occ)
    if params.method == "visibility":
        return wrap_visibility(bg, alpha, band, params, occ)
    raise ValueError(f"未知包光方法: {params.method}")
