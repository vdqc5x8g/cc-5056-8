import numpy as np


def composite_base(fg, alpha, bg):
    """标准透明混合(线性域): 已包含一份背景贡献."""
    a = alpha[:, :, None]
    return fg * a + bg * (1.0 - a)


def composite_with_wrap(fg, alpha, bg, wrap_layer):
    """叠加包光层.

    wrap_layer 只含周边辐射超出局部背景的部分, 与 composite_base 中
    已有的背景项不重复 —— 避免重复计入背景造成过亮轮廓.
    """
    return composite_base(fg, alpha, bg) + wrap_layer
