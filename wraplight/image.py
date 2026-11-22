import numpy as np
from PIL import Image


def load_rgb(path):
    """读取 sRGB 图像为 HxWx3 float64, 0..1."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def load_alpha(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def load_mask(path):
    img = Image.open(path).convert("L")
    return (np.asarray(img, dtype=np.float64) / 255.0 > 0.5).astype(np.float64)


def save_rgb(path, img):
    """保存 0..1 float 图像为 8bit PNG(调用方负责色彩空间转换)."""
    arr = np.clip(img, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), "RGB").save(path)


def save_gray(path, img):
    arr = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), "L").save(path)


def save_heatmap(path, img):
    """把单通道标量图渲染为伪彩, 便于查看包光层/可见性分布."""
    v = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    r = np.clip(1.5 * v - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * v - 1.5), 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * v, 0.0, 1.0)
    save_rgb(path, np.stack([r, g, b], axis=-1))
