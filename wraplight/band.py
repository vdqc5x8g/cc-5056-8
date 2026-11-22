from dataclasses import dataclass

import numpy as np

from .blur import gaussian_blur
from .distance import signed_distance


@dataclass
class EdgeBand:
    """轮廓附近的包光作用范围.

    weight:   HxW, 0..1, 带内权重(向两侧渐隐)
    normal:   HxWx2, 指向背景侧的单位向量 (dy, dx)
    signed:   HxW, 相对轮廓的符号距离, 前景内部为正
    """

    weight: np.ndarray
    normal: np.ndarray
    signed: np.ndarray


def _normals_from_signed(signed):
    smoothed = gaussian_blur(signed, 1.5)
    gy, gx = np.gradient(smoothed)
    mag = np.hypot(gy, gx)
    mag = np.where(mag < 1e-9, 1.0, mag)
    # signed 向前景内部增大, 负梯度方向即背景侧
    return np.stack([-gy / mag, -gx / mag], axis=-1)


def band_from_alpha(alpha, inner_width, outer_width, level=0.5):
    """由透明度定义边缘范围: 距轮廓 inner/outer 宽度内的像素."""
    signed = signed_distance(alpha, level)
    inner = np.clip(1.0 - signed / max(inner_width, 1e-6), 0.0, 1.0)
    outer = np.clip(1.0 + signed / max(outer_width, 1e-6), 0.0, 1.0)
    weight = np.where(signed >= 0.0, inner, outer)
    return EdgeBand(weight=weight, normal=_normals_from_signed(signed), signed=signed)


def band_from_mask(mask, alpha):
    """由明确的几何 mask 定义边缘范围."""
    weight = np.asarray(mask, dtype=np.float64)
    if weight.shape != alpha.shape:
        raise ValueError("几何 mask 与 alpha 尺寸不一致")
    signed = signed_distance(alpha, 0.5)
    return EdgeBand(
        weight=np.clip(weight, 0.0, 1.0),
        normal=_normals_from_signed(signed),
        signed=signed,
    )
