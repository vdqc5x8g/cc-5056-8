import math

import numpy as np


def box_blur(img, radius):
    radius = int(round(radius))
    if radius <= 0:
        return np.asarray(img, dtype=np.float64).copy()
    img = np.asarray(img, dtype=np.float64)
    pad_width = ((radius, radius), (radius, radius))
    if img.ndim == 3:
        pad_width = pad_width + ((0, 0),)
    padded = np.pad(img, pad_width, mode="edge")
    integ = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    zero_pad = ((1, 0), (1, 0)) + (((0, 0),) if img.ndim == 3 else ())
    integ = np.pad(integ, zero_pad, mode="constant")
    w = 2 * radius + 1
    total = (
        integ[w:, w:]
        - integ[:-w, w:]
        - integ[w:, :-w]
        + integ[:-w, :-w]
    )
    return total / float(w * w)


def _gauss_box_widths(sigma, count=3):
    w_ideal = math.sqrt(12.0 * sigma * sigma / count + 1.0)
    wl = int(math.floor(w_ideal))
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 1)
    m = (12.0 * sigma * sigma - count * wl * wl - 4 * count * wl - 3 * count) / (
        -4 * wl - 4
    )
    m = int(round(m))
    m = min(max(m, 0), count)
    return [wl if i < m else wl + 2 for i in range(count)]


def gaussian_blur(img, sigma):
    sigma = float(sigma)
    if sigma < 0.5:
        return np.asarray(img, dtype=np.float64).copy()
    out = np.asarray(img, dtype=np.float64)
    for width in _gauss_box_widths(sigma):
        out = box_blur(out, (width - 1) // 2)
    return out
