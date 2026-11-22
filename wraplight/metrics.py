"""包光执行指标: 边缘能量 / 主体色差 / 边缘融合.

所有数值由代码从运行结果复算, 不写手工估计值; 每个指标对应可回查的
执行证据(图层 PNG 与 params.json 随运行目录保存).
"""

import numpy as np


def edge_energy(wrap_layer, band_weight):
    """边缘带内包光层能量(线性域 RGB 模长)的均值/峰值."""
    energy = np.linalg.norm(np.asarray(wrap_layer, dtype=np.float64), axis=-1)
    band = np.asarray(band_weight) > 1e-3
    if not band.any():
        return {"mean": 0.0, "max": 0.0, "band_pixels": 0}
    return {
        "mean": float(energy[band].mean()),
        "max": float(energy[band].max()),
        "band_pixels": int(band.sum()),
    }


def subject_color_diff(composite_on, fg_linear, signed_dist, inner_width):
    """主体色差: 合成结果在主体内部相对线性前景的平均色差.

    内部颜色保留的量化 —— 包光只允许调整轮廓附近, 该值应接近 0;
    整体模糊等错误替代会把背景渗进内部, 该值显著增大.
    """
    inside = np.asarray(signed_dist) > (inner_width + 1.0)
    if not inside.any():
        return {"mean": 0.0, "max": 0.0, "interior_pixels": 0}
    diff = np.linalg.norm(
        np.asarray(composite_on, dtype=np.float64)
        - np.asarray(fg_linear, dtype=np.float64),
        axis=-1,
    )
    return {
        "mean": float(diff[inside].mean()),
        "max": float(diff[inside].max()),
        "interior_pixels": int(inside.sum()),
    }


def edge_blend_diff(composite_on, composite_off, band_weight):
    """边缘融合: 包光开/关合成结果在边缘带内的平均色差."""
    band = np.asarray(band_weight) > 1e-3
    if not band.any():
        return {"mean": 0.0, "max": 0.0}
    diff = np.linalg.norm(
        np.asarray(composite_on, dtype=np.float64)
        - np.asarray(composite_off, dtype=np.float64),
        axis=-1,
    )
    return {
        "mean": float(diff[band].mean()),
        "max": float(diff[band].max()),
    }


def wrap_metrics(out, fg_linear):
    """对一次流水线输出复算全部指标(内部颜色保留与边缘融合分列)."""
    inner = out.params.inner_width if out.params is not None else 6.0
    return {
        "edge_energy": edge_energy(out.wrap_layer, out.band_weight),
        "subject_color_diff": subject_color_diff(
            out.composite_on, fg_linear, out.signed_dist, inner),
        "edge_blend": edge_blend_diff(
            out.composite_on, out.composite_off, out.band_weight),
    }
