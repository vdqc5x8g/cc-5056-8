from dataclasses import dataclass, field

import numpy as np

from .band import band_from_alpha, band_from_mask
from .colorspace import linear_to_srgb, srgb_to_linear
from .composite import composite_base, composite_with_wrap
from .deformer import apply_deformer
from .wrap import compute_wrap


@dataclass
class PipelineOutput:
    composite_on: np.ndarray      # 包光开启的合成结果(线性)
    composite_off: np.ndarray     # 包光关闭的合成结果(线性)
    wrap_layer: np.ndarray        # 包光层(线性, 已定位到发丝)
    wrap_raw: np.ndarray          # 变形器之前的包光层
    excess: np.ndarray            # 周边超出局部背景的部分
    band_weight: np.ndarray       # 边缘带权重
    signed_dist: np.ndarray       # 符号距离
    debug: dict = field(default_factory=dict)
    params: object = None


def apply_occluder_to_bg(bg, occluder, plate_color):
    """把挡板作为实体合成进背景: 挡板区域显示其自身颜色(与发丝同色)."""
    if occluder is None:
        return bg
    occ = np.asarray(occluder, dtype=np.float64)[:, :, None]
    plate = np.asarray(plate_color, dtype=np.float64).reshape(1, 1, 3)
    return bg * (1.0 - occ) + plate * occ


def run_pipeline(fg, alpha, bg, params, occluder=None, geom_mask=None):
    """完整流水线.

    fg/bg 为 sRGB 0..1 (bg 允许 >1 的 HDR 值), alpha 0..1.
    linear=True 时全程在线性光度下计算; False 时直接在 sRGB 域计算
    (仅用于对照展示重复计入/过亮问题).
    """
    if params.linear:
        fg_c = srgb_to_linear(np.clip(fg, 0.0, 1.0))
        bg_c = srgb_to_linear(np.clip(bg, 0.0, None))
    else:
        fg_c = np.clip(fg, 0.0, 1.0).astype(np.float64)
        bg_c = np.clip(bg, 0.0, None).astype(np.float64)
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0)

    if geom_mask is not None:
        band = band_from_mask(geom_mask, alpha)
    else:
        band = band_from_alpha(alpha, params.inner_width, params.outer_width)

    result = compute_wrap(bg_c, alpha, band, params, occluder)
    wrap_layer = apply_deformer(result.layer, alpha, params)

    off = composite_base(fg_c, alpha, bg_c)
    on = composite_with_wrap(fg_c, alpha, bg_c, wrap_layer)
    return PipelineOutput(
        composite_on=on,
        composite_off=off,
        wrap_layer=wrap_layer,
        wrap_raw=result.layer,
        excess=result.excess,
        band_weight=band.weight,
        signed_dist=band.signed,
        debug=result.debug,
        params=params,
    )


def to_display(img_linear, params):
    """线性结果显示为 sRGB; 非线性模式直接返回."""
    if params is not None and not params.linear:
        return np.clip(img_linear, 0.0, 1.0)
    return linear_to_srgb(img_linear)


def edge_check_crop(img, band_weight, pad=8, scale=4):
    """回查替换背景后的真实边缘: 裁出边缘带包围盒并最近邻放大."""
    ys, xs = np.nonzero(band_weight > 1e-3)
    if ys.size == 0:
        return img
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.shape[1])
    crop = img[y0:y1, x0:x1]
    return np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)
