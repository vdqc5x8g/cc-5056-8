import numpy as np

from .blur import gaussian_blur


def sinusoid_field(shape, amplitude, frequency):
    """内置位移场: 模拟前景(发丝)提取后经历的轻微形变."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = amplitude * np.sin(2.0 * np.pi * frequency * yy / max(h, 1) * 8.0)
    dy = amplitude * np.cos(2.0 * np.pi * frequency * xx / max(w, 1) * 8.0)
    return dx, dy


def warp_layer(layer, dx, dy):
    """按位移场反向采样, 把包光贡献重新定位到变形后的发丝上."""
    h, w = layer.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sy = np.clip(yy - dy, 0.0, h - 1.0)
    sx = np.clip(xx - dx, 0.0, w - 1.0)
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (sy - y0)[:, :, None]
    fx = (sx - x0)[:, :, None]
    return (
        layer[y0, x0] * (1 - fy) * (1 - fx)
        + layer[y0, x1] * (1 - fy) * fx
        + layer[y1, x0] * fy * (1 - fx)
        + layer[y1, x1] * fy * fx
    )


def strand_gate(alpha, sigma=1.5):
    """发丝定位门控: alpha 的高频细节刻画发丝等细碎结构."""
    base = gaussian_blur(alpha, sigma)
    detail = np.clip(alpha - base, 0.0, 1.0)
    peak = detail.max()
    if peak > 1e-6:
        detail = detail / peak
    return detail


def apply_deformer(wrap_layer, alpha, params):
    """前景变形器: 位移场重定位 + 发丝门控, 把贡献定位到发丝上."""
    out = wrap_layer
    if params.deform_amp > 1e-6:
        dx, dy = sinusoid_field(alpha.shape, params.deform_amp, params.deform_freq)
        out = warp_layer(out, dx, dy)
    if params.strand_strength > 1e-6:
        gate = strand_gate(alpha)
        s = min(max(params.strand_strength, 0.0), 1.0)
        out = out * ((1.0 - s) + s * gate)[:, :, None]
    return out
