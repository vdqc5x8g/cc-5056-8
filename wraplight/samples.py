"""样本生成: 前景细枝、硬边、背景局部亮源、与发丝同色的挡板."""

import math

import numpy as np

from .blur import gaussian_blur
from .image import save_gray, save_rgb

HAIR_COLOR = (0.16, 0.10, 0.06)   # 发丝(与挡板同色), sRGB
LEAF_COLOR = (0.10, 0.24, 0.08)


def _draw_line(mask, x0, y0, x1, y1, radius):
    h, w = mask.shape
    length = max(math.hypot(x1 - x0, y1 - y0), 1.0)
    steps = int(length * 2) + 1
    for t in np.linspace(0.0, 1.0, steps):
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        r = radius * (1.0 - 0.6 * t) + 0.4
        x0i = max(int(x - r - 1), 0)
        x1i = min(int(x + r + 2), w)
        y0i = max(int(y - r - 1), 0)
        y1i = min(int(y + r + 2), h)
        yy, xx = np.mgrid[y0i:y1i, x0i:x1i]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        mask[y0i:y1i, x0i:x1i] = np.maximum(
            mask[y0i:y1i, x0i:x1i], (d2 <= r * r).astype(np.float64)
        )


def make_fine_branches(size=256, seed=7):
    """细枝前景: 1~3px 分叉枝条, 模拟发丝级细碎结构."""
    rng = np.random.default_rng(seed)
    alpha = np.zeros((size, size), dtype=np.float64)

    def branch(x, y, angle, length, radius, depth):
        if depth == 0 or length < 4:
            return
        x1 = x + math.cos(angle) * length
        y1 = y + math.sin(angle) * length
        _draw_line(alpha, x, y, x1, y1, radius)
        n = 2 if depth > 2 else 1
        for _ in range(n):
            spread = rng.uniform(0.25, 0.7)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            branch(
                x1, y1,
                angle + sign * spread,
                length * rng.uniform(0.62, 0.8),
                radius * 0.7,
                depth - 1,
            )

    branch(size * 0.5, size * 0.98, -math.pi / 2, size * 0.24, 2.6, 5)
    alpha = np.clip(gaussian_blur(alpha, 0.6), 0.0, 1.0)
    fg = np.zeros((size, size, 3), dtype=np.float64)
    fg[:] = HAIR_COLOR
    tint = np.clip(gaussian_blur(alpha, 2.0) - alpha, 0.0, 1.0)[:, :, None]
    fg = fg * (1 - 0.35 * tint) + np.array(LEAF_COLOR) * 0.35 * tint
    return fg, alpha


def make_hard_edge(size=256):
    """硬边前景: 锐利的矩形+圆盘, alpha 无过渡."""
    alpha = np.zeros((size, size), dtype=np.float64)
    s = size
    alpha[int(s * 0.30):int(s * 0.78), int(s * 0.30):int(s * 0.62)] = 1.0
    yy, xx = np.mgrid[0:s, 0:s]
    alpha[(yy - s * 0.52) ** 2 + (xx - s * 0.68) ** 2 <= (s * 0.13) ** 2] = 1.0
    fg = np.zeros((s, s, 3), dtype=np.float64)
    fg[:] = (0.32, 0.20, 0.42)
    fg[int(s * 0.42):int(s * 0.58), int(s * 0.34):int(s * 0.58)] = (0.55, 0.38, 0.10)
    return fg, alpha


def make_background(size=256, bright_center=(0.30, 0.22), bright_radius=0.10):
    """新背景: 渐变 + 局部亮源(亮盘) + 低频纹理."""
    s = size
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float64) / s
    bg = np.zeros((s, s, 3), dtype=np.float64)
    bg[..., 0] = 0.18 + 0.25 * (1 - yy)
    bg[..., 1] = 0.22 + 0.30 * (1 - yy)
    bg[..., 2] = 0.30 + 0.35 * (1 - yy)
    tex = gaussian_blur(
        np.random.default_rng(3).random((s, s, 3)), 6.0
    )
    bg = bg * (0.85 + 0.3 * tex)
    cx, cy = bright_center[0] * s, bright_center[1] * s
    d2 = (xx * s - cx) ** 2 + (yy * s - cy) ** 2
    disc = np.clip(1.0 - np.sqrt(d2) / (bright_radius * s), 0.0, 1.0)
    disc = np.clip(disc * 1.4, 0.0, 1.0)
    bg += disc[:, :, None] * np.array([2.6, 2.3, 1.7])
    return np.clip(bg, 0.0, 3.5)


def make_occluder(size=256, box=(0.34, 0.30, 0.44, 0.62)):
    """与发丝同色的挡板 mask; box=(x0, y0, x1, y1) 归一化坐标."""
    s = size
    occ = np.zeros((s, s), dtype=np.float64)
    x0, y0, x1, y1 = box
    occ[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = 1.0
    return occ


def make_occluder_plate_image(size=256, box=(0.34, 0.30, 0.44, 0.62)):
    """挡板贴到背景上的颜色(与发丝同色), 用于演示遮后颜色."""
    s = size
    plate = np.zeros((s, s, 3), dtype=np.float64)
    plate[:] = HAIR_COLOR
    return plate, make_occluder(size, box)


def generate_samples(out_dir, size=256):
    import os

    os.makedirs(out_dir, exist_ok=True)
    fg_b, a_b = make_fine_branches(size)
    fg_h, a_h = make_hard_edge(size)
    bg = make_background(size)
    plate, occ_near = make_occluder_plate_image(size)
    occ_far = make_occluder(size, box=(0.60, 0.30, 0.70, 0.62))

    save_rgb(os.path.join(out_dir, "fg_branches.png"), fg_b)
    save_gray(os.path.join(out_dir, "alpha_branches.png"), a_b)
    save_rgb(os.path.join(out_dir, "fg_hard.png"), fg_h)
    save_gray(os.path.join(out_dir, "alpha_hard.png"), a_h)
    save_rgb(os.path.join(out_dir, "background.png"), np.clip(bg, 0.0, 1.0))
    save_gray(os.path.join(out_dir, "occluder_near.png"), occ_near)
    save_gray(os.path.join(out_dir, "occluder_far.png"), occ_far)
    save_rgb(os.path.join(out_dir, "plate_color.png"), plate)
    return out_dir
