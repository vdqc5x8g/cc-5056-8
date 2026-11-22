import math

import numpy as np

_INF = 1e18
_SQRT2 = math.sqrt(2.0)


def chamfer_distance(target):
    """近似欧氏距离变换: 每个像素到最近 True 像素的距离."""
    target = np.asarray(target, dtype=bool)
    h, w = target.shape
    dist = np.where(target, 0.0, _INF)
    d = dist.tolist()
    for y in range(h):
        row = d[y]
        row_up = d[y - 1] if y > 0 else None
        for x in range(w):
            v = row[x]
            if v == 0.0:
                continue
            best = v
            if x > 0:
                best = min(best, row[x - 1] + 1.0)
            if row_up is not None:
                best = min(best, row_up[x] + 1.0)
                if x > 0:
                    best = min(best, row_up[x - 1] + _SQRT2)
                if x + 1 < w:
                    best = min(best, row_up[x + 1] + _SQRT2)
            row[x] = best
    for y in range(h - 1, -1, -1):
        row = d[y]
        row_dn = d[y + 1] if y + 1 < h else None
        for x in range(w - 1, -1, -1):
            v = row[x]
            if v == 0.0:
                continue
            best = v
            if x + 1 < w:
                best = min(best, row[x + 1] + 1.0)
            if row_dn is not None:
                best = min(best, row_dn[x] + 1.0)
                if x + 1 < w:
                    best = min(best, row_dn[x + 1] + _SQRT2)
                if x > 0:
                    best = min(best, row_dn[x - 1] + _SQRT2)
            row[x] = best
    return np.asarray(d, dtype=np.float64)


def signed_distance(alpha, level=0.5):
    """相对 alpha=level 轮廓的符号距离, 前景内部为正."""
    inside = alpha >= level
    d_out = chamfer_distance(~inside)
    d_in = chamfer_distance(inside)
    return np.where(inside, d_out, -d_in)
