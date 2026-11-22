import numpy as np

_SRGB_THRESHOLD = 0.04045
_LINEAR_THRESHOLD = 0.0031308


def srgb_to_linear(img):
    img = np.asarray(img, dtype=np.float64)
    return np.where(
        img <= _SRGB_THRESHOLD,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(img):
    img = np.clip(np.asarray(img, dtype=np.float64), 0.0, None)
    return np.where(
        img <= _LINEAR_THRESHOLD,
        img * 12.92,
        1.055 * np.power(img, 1.0 / 2.4) - 0.055,
    )
