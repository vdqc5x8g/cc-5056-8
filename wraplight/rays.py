from dataclasses import dataclass, field

import numpy as np


@dataclass
class VisibilityResult:
    """遮挡求值器输出: 边缘带像素对背景亮源的可见采样积分."""

    integral: np.ndarray          # HxWx3 可见背景辐射的加权平均(线性)
    visibility: np.ndarray        # HxW  marched 权重中属于可见背景的比例
    blocked: np.ndarray           # HxW 被遮挡物截停的射线方向权重比例
    mean_hit_dist: np.ndarray     # HxW 背景样本的平均距离
    band_pixels: tuple = field(default=None)


def integrate_visible_background(
    bg,
    alpha,
    band,
    occluder=None,
    ray_count=16,
    max_dist=48.0,
    cone_angle=1.2,
    falloff_tau=24.0,
    bg_alpha_threshold=0.05,
    use_occlusion=True,
):
    """对边缘带内每个像素, 朝背景侧的锥形范围发射射线并沿射线积分.

    射线每步进 1px: 穿过前景(alpha 高于阈值)时不贡献但累计距离;
    经过可见背景段时按 方向余弦 x 距离衰减 累计其辐射 —— 亮源的
    可见片段被送入包光; 命中遮挡物的射线采样挡板表面辐射后被截停,
    其后的亮源不再贡献 —— 亮物移到遮挡面后即停止包光.
    use_occlusion=False 时退化为纯距离衰减估算(不查遮挡), 用于与
    真实可见背景估算对照.

    积分按累计权重归一化: 均匀背景下积分恒等于该背景色, 包光超出
    量为零, 不会重复计入背景.
    """
    h, w = alpha.shape
    ys, xs = np.nonzero(band.weight > 1e-4)
    n_px = ys.size
    integral = np.zeros((h, w, 3), dtype=np.float64)
    visibility = np.zeros((h, w), dtype=np.float64)
    blocked = np.zeros((h, w), dtype=np.float64)
    mean_dist = np.zeros((h, w), dtype=np.float64)
    if n_px == 0:
        return VisibilityResult(integral, visibility, blocked, mean_dist, (ys, xs))

    normals = band.normal[ys, xs]                      # (N,2) -> (dy, dx)
    base_ang = np.arctan2(normals[:, 0], normals[:, 1])
    k = max(int(ray_count), 1)
    if k == 1:
        offsets = np.zeros(1)
    else:
        offsets = np.linspace(-cone_angle, cone_angle, k)
    dir_w = np.cos(offsets)                            # 方向余弦权重 (K,)
    dir_w_sum = float(dir_w.sum())

    ang = base_ang[:, None] + offsets[None, :]         # (N,K)
    step_dy = np.sin(ang)
    step_dx = np.cos(ang)

    pos_y = np.broadcast_to(ys[:, None].astype(np.float64), (n_px, k)).copy()
    pos_x = np.broadcast_to(xs[:, None].astype(np.float64), (n_px, k)).copy()
    alive = np.ones((n_px, k), dtype=bool)
    acc = np.zeros((n_px, k, 3), dtype=np.float64)     # 背景辐射加权和
    acc_w = np.zeros((n_px, k), dtype=np.float64)      # 背景样本权重和
    tot_w = np.zeros((n_px, k), dtype=np.float64)      # marched 权重和(含前景段)
    blocked_w = np.zeros((n_px, k), dtype=np.float64)
    dist_acc = np.zeros((n_px, k), dtype=np.float64)

    occ = None
    if use_occlusion and occluder is not None:
        occ = np.asarray(occluder) > 0.5

    dir_w_full = np.broadcast_to(dir_w, (n_px, k))
    max_steps = int(max_dist)
    for step in range(1, max_steps + 1):
        if not alive.any():
            break
        pos_y[alive] += step_dy[alive]
        pos_x[alive] += step_dx[alive]
        iy = np.round(pos_y).astype(np.int64)
        ix = np.round(pos_x).astype(np.int64)
        out = (iy < 0) | (iy >= h) | (ix < 0) | (ix >= w)
        alive &= ~out
        if not alive.any():
            continue
        decay = np.exp(-step / max(falloff_tau, 1e-6))
        w_step = dir_w_full * decay
        tot_w[alive] += w_step[alive]
        if occ is not None:
            hit_occ = np.zeros((n_px, k), dtype=bool)
            hit_occ[alive] = occ[iy[alive], ix[alive]]
            if hit_occ.any():
                # 挡板是可见表面: 采样其辐射(背景图中挡板颜色)后截停
                w_sel = w_step[hit_occ]
                acc[hit_occ] += bg[iy[hit_occ], ix[hit_occ]] * w_sel[:, None]
                acc_w[hit_occ] += w_sel
                dist_acc[hit_occ] += step * w_sel
                blocked_w[hit_occ] += dir_w_full[hit_occ]
                alive &= ~hit_occ
            if not alive.any():
                continue
        a = np.ones((n_px, k))
        a[alive] = alpha[iy[alive], ix[alive]]
        on_bg = (a <= bg_alpha_threshold) & alive
        if on_bg.any():
            cols = np.zeros((n_px, k, 3))
            cols[on_bg] = bg[iy[on_bg], ix[on_bg]]
            w_sel = w_step[on_bg]
            acc[on_bg] += cols[on_bg] * w_sel[:, None]
            acc_w[on_bg] += w_sel
            dist_acc[on_bg] += step * w_sel

    w_sum = acc_w.sum(axis=1)
    has = w_sum > 1e-12
    integral[ys[has], xs[has]] = (
        acc[has].sum(axis=1) / w_sum[has, None]
    )
    visibility[ys, xs] = acc_w.sum(axis=1) / np.maximum(tot_w.sum(axis=1), 1e-12)
    blocked[ys, xs] = blocked_w.sum(axis=1) / max(dir_w_sum, 1e-9)
    mean_dist[ys[has], xs[has]] = (
        dist_acc[has].sum(axis=1) / w_sum[has]
    )
    return VisibilityResult(integral, visibility, blocked, mean_dist, (ys, xs))
