import json
from dataclasses import asdict, dataclass, fields


@dataclass
class WrapParams:
    """包光合成参数, 可序列化为 JSON 保存/复现."""

    method: str = "visibility"   # blur | falloff | visibility | depth_select | vis_mix | switch
    inner_width: float = 6.0     # 边缘带向前景内部宽度(px)
    outer_width: float = 10.0    # 边缘带向背景侧宽度(px)
    intensity: float = 1.0       # 包光整体强度
    blur_sigma: float = 12.0     # blur 法: 背景模糊半径
    dir_offset: float = 8.0      # blur 法: 朝背景侧的采样偏移(px), 方向限制
    ray_count: int = 16          # 射线法: 每像素射线数
    max_dist: float = 48.0       # 射线法: 最大采样距离(px), 距离限制
    cone_angle: float = 1.2      # 射线法: 方向锥半角(rad)
    falloff_tau: float = 24.0    # 射线法: 距离衰减尺度(px)
    allow_negative: bool = False  # 允许负超出量(周边比局部暗时压暗边缘)
    deform_amp: float = 0.0      # 前景变形器: 位移场幅度(px)
    deform_freq: float = 0.12    # 前景变形器: 位移场频率
    strand_strength: float = 0.0  # 前景变形器: 发丝门控强度 0..1
    linear: bool = True          # 线性光度开关(关闭则在 sRGB 域计算, 仅用于对照)
    use_occluder: bool = True    # 是否启用遮挡物输入
    # 组合贡献模型的判定门槛: 默认 None, 用到时必须显式给出 —— 不以默认值遮错
    depth_threshold: float = None  # depth_select: 局部深度(平均命中距离)门槛(px)
    occ_threshold: float = None    # switch: 遮挡比例门槛(0..1), 超过则切到方向加权

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
