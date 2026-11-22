# wraplight — 前景边缘包光合成工具

从已确认前景(fg+alpha)与新背景出发, 估算前景轮廓附近的包光贡献并合成。
包光只调整轮廓附近的融合, 不改变主体内部; 不是完整真实光照模拟。

## 环境

Python 3.11+, 依赖: numpy, pillow; 测试: pytest。

## 界面怎样启动

```bash
python3 -m wraplight serve --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

界面控件(估算方法、各参数滑条)由边缘包光内核经 `/api/controls` 下发,
前端不硬编码; 组合模型的判定门槛(深度门槛/遮挡门槛)无默认值, 需在界面
勾选启用并显式设置, 否则内核报错并反馈到界面报告区。

画布视图: 合成·包光开 / 合成·包光关 / 轮廓对照(关|开并排) / 包光层 /
边缘带 / 可见性 / 遮挡比例 / 超出量 / 边缘回查。调节包光范围(带内宽/
带外宽)后, 用"轮廓对照"与"合成·包光开"对照轮廓亮边与合成结果。
报告区显示边缘能量、主体色差(内部颜色保留)与边缘融合三项复算指标。

## 命令行

```bash
python3 -m wraplight sample                 # 生成样本到 samples/
python3 -m wraplight run --fg samples/fg_branches.png \
    --alpha samples/alpha_branches.png --bg samples/background.png \
    --set method=visibility --out outputs/run
python3 -m wraplight compare                # 挡板对照(旧样例)
python3 -m wraplight experiment             # 全部包光模型实验
python3 -m wraplight experiment --name occlusion_switch
python3 -m wraplight limitations            # 近似/伪影/待验假设
python3 -m pytest tests/                    # 测试
```

## 贡献模型(可切换)

- `blur` 局部背景平均: 模糊背景 + 朝背景侧偏移采样, 不查遮挡。
- `falloff` / `visibility` 方向加权: 锥形射线积分可见背景, 遮挡截停。
- `depth_select` 按局部深度选择(参照): 近处方向加权, 远处局部平均;
  需显式 `depth_threshold`。
- `vis_mix` 按可见背景份额混合两种估算。
- `switch` 遮挡门槛双模型切换; 需显式 `occ_threshold`。

## 包光模型实验

`python3 -m wraplight experiment` 运行五个实验(model_compare /
single_side / fine_structure / nonuniform_bg / occlusion_switch),
每层证据(合成图、包光层、可见性、params.json)写入
`outputs/experiments/<name>/`, 指标由代码复算进 `report.json`,
结论汇总于 `outputs/experiments/CONCLUSIONS.md`(数字来自复算, 不手写)。
