"""贡献模型: 局部背景平均与方向加权两种基础估算的组合方案.

基础模型(见 wrap.py):
  blur        基于局部背景平均: 模糊背景 + 朝背景侧偏移采样, 不查遮挡.
  visibility  方向加权: 朝背景侧锥形射线积分, 方向余弦加权, 遮挡截停.

组合模型(本模块):
  depth_select  按局部深度选择包光(参照): 平均命中距离 <= depth_threshold
                的像素用方向加权, 其余用局部背景平均.
  vis_mix       按可见背景份额混合: excess = v*方向加权 + (1-v)*局部平均,
                v 为可见背景份额; 适用范围由实验检验(见 experiments.py).
  switch        遮挡门槛双模型切换: 遮挡比例 > occ_threshold 的像素切到
                方向加权, 其余保持局部背景平均.

判定门槛(depth_threshold / occ_threshold)没有默认值: 用到而未显式给出时
抛出 ValueError, 不以默认值遮错.
"""

import numpy as np

METHODS = ("blur", "falloff", "visibility", "depth_select", "vis_mix", "switch")
COMBINED_METHODS = ("depth_select", "vis_mix", "switch")


def require_threshold(params, name):
    """读取判定门槛; 未显式设置即报错, 不允许默认值掩盖配置缺失."""
    value = getattr(params, name)
    if value is None:
        raise ValueError(
            f"方法 {params.method} 需要显式设置 {name} "
            f"(当前为 None; 包光模型实验不得以默认值遮错)"
        )
    return float(value)


def combine_depth_select(excess_dir, excess_local, mean_hit_dist, depth_threshold):
    """按局部深度逐像素选择: 近处背景用方向加权, 远处用局部平均."""
    near = np.asarray(mean_hit_dist) <= float(depth_threshold)
    sel = near[:, :, None]
    return np.where(sel, excess_dir, excess_local), near


def combine_vis_mix(excess_dir, excess_local, visibility):
    """按可见背景份额混合两种估算."""
    v = np.clip(np.asarray(visibility, dtype=np.float64), 0.0, 1.0)[:, :, None]
    return v * excess_dir + (1.0 - v) * excess_local


def combine_switch(excess_dir, excess_local, blocked, occ_threshold):
    """遮挡门槛双模型切换: 遮挡比例超门槛的像素切到方向加权."""
    use_dir = np.asarray(blocked) > float(occ_threshold)
    sel = use_dir[:, :, None]
    return np.where(sel, excess_dir, excess_local), use_dir


def kernel_controls():
    """边缘包光内核暴露的控件规格 —— UI 控件由内核驱动, 不在前端硬编码.

    每项: (参数名, 标签, 最小, 最大, 步长, 适用方法元组).
    门槛类控件无默认值: default=None 表示必须由调用方显式给出.
    """
    ray = ("falloff", "visibility", "depth_select", "vis_mix", "switch")
    local = ("blur", "depth_select", "vis_mix", "switch")
    return {
        "methods": [
            {"id": "visibility", "label": "方向加权·可见区域积分(遮挡)"},
            {"id": "falloff", "label": "方向加权·仅距离衰减(对照)"},
            {"id": "blur", "label": "局部背景平均(模糊近似)"},
            {"id": "depth_select", "label": "组合·按局部深度选择(参照)"},
            {"id": "vis_mix", "label": "组合·按可见背景份额混合"},
            {"id": "switch", "label": "组合·遮挡门槛双模型切换"},
        ],
        "sliders": [
            {"key": "inner_width", "label": "带内宽(px)", "min": 0, "max": 32,
             "step": 0.5, "default": 6.0, "methods": None},
            {"key": "outer_width", "label": "带外宽(px)", "min": 0, "max": 48,
             "step": 0.5, "default": 10.0, "methods": None},
            {"key": "intensity", "label": "强度", "min": 0, "max": 3,
             "step": 0.05, "default": 1.0, "methods": None},
            {"key": "blur_sigma", "label": "局部平均半径", "min": 1, "max": 48,
             "step": 0.5, "default": 12.0, "methods": local},
            {"key": "dir_offset", "label": "方向偏移(px)", "min": 0, "max": 32,
             "step": 0.5, "default": 8.0, "methods": ("blur",)},
            {"key": "ray_count", "label": "射线数", "min": 1, "max": 64,
             "step": 1, "default": 16, "methods": ray},
            {"key": "max_dist", "label": "最大距离(px)", "min": 4, "max": 128,
             "step": 1, "default": 48.0, "methods": ray},
            {"key": "cone_angle", "label": "方向锥半角(rad)", "min": 0.1,
             "max": 3.14, "step": 0.01, "default": 1.2, "methods": ray},
            {"key": "falloff_tau", "label": "距离衰减(px)", "min": 2, "max": 96,
             "step": 1, "default": 24.0, "methods": ray},
            {"key": "depth_threshold", "label": "深度门槛(px)", "min": 1,
             "max": 128, "step": 1, "default": None,
             "methods": ("depth_select",)},
            {"key": "occ_threshold", "label": "遮挡门槛(0..1)", "min": 0.0,
             "max": 1.0, "step": 0.01, "default": None,
             "methods": ("switch",)},
            {"key": "deform_amp", "label": "变形幅度(px)", "min": 0, "max": 8,
             "step": 0.1, "default": 0.0, "methods": None},
            {"key": "deform_freq", "label": "变形频率", "min": 0.01, "max": 0.5,
             "step": 0.01, "default": 0.12, "methods": None},
            {"key": "strand_strength", "label": "发丝门控", "min": 0, "max": 1,
             "step": 0.05, "default": 0.0, "methods": None},
        ],
    }


MODEL_DISCUSSION = {
    "local_avg": (
        "局部背景平均(blur): 用模糊后的邻域平均近似周边辐射, 方向偏移采样"
        " 提供粗略方向性. 不查询遮挡 —— 亮物被挡后仍按原背景包光; 半径过大"
        " 会把远处亮结构晕进边缘, 非均匀背景下偏差明显."
    ),
    "dir_weighted": (
        "方向加权(visibility): 锥形射线按方向余弦加权、距离衰减积分可见"
        " 背景, 遮挡截停射线. 对单侧亮源与遮挡变化敏感, 但细发丝(<2px)处"
        " 法线抖动会放大方向权重噪声."
    ),
    "depth_select": (
        "按局部深度选择(参照): 近处背景用方向加权、远处用局部平均. 作为"
        " 检验其他组合方案的参照 —— 它不混合两种估算, 只按深度切换."
    ),
    "vis_mix": (
        "按可见背景份额混合: 可见份额高时偏向方向加权, 低时偏向局部平均."
        " 遮挡边缘附近过渡平滑, 但遮挡锐变时混合区会出现两种估算的折中色,"
        " 适用范围由 occlusion_switch / nonuniform_bg 实验检验."
    ),
    "switch": (
        "遮挡门槛双模型切换: 遮挡比例超门槛即整体切到方向加权. 背景亮度"
        " 不变而遮挡改变时, 变化方向与参照相反或幅度不足的模型(如局部平均)"
        " 在该区域被淘汰."
    ),
}

# 贡献模型讨论的假设 —— 单列待验, 由 experiments.py 中的实验逐一核对.
PENDING_ASSUMPTIONS = [
    "细发丝(<2px)上方向加权法线噪声可被可见份额混合(vis_mix)平均 —— 待验.",
    "大轮廓硬边上局部背景平均与方向加权的边缘融合差异主要来自方向性, "
    "而非半径 —— 待验.",
    "非均匀背景下按局部深度选择(depth_select)可代表两种基础模型的上界 "
    "—— 待验(目前仅作参照, 未与更优组合比较).",
    "遮挡门槛切换的临界值对前景尺度不敏感 —— 待验(当前实验仅 256px 样本).",
    "整体模糊不能替代包光: 其主体色差显著大于任何包光模型 —— 已在 "
    "fine_structure 实验中核对, 仍列为假设以跟踪新样本.",
]
