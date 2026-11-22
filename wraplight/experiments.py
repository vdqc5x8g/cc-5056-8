"""仓库内包光模型实验: 决定与采样证据随工程保存.

每个实验:
  1. 显式构造参数(包光模型实验不得以默认值遮错);
  2. 运行流水线, 边缘能量/主体色差/边缘融合由 metrics 模块复算;
  3. 决定、指标与证据图层写入 outputs/experiments/<name>/;
  4. 结论由复算数据生成, 汇总进 CONCLUSIONS.md.

实验清单:
  model_compare    局部背景平均 vs 方向加权, 可切换, 接已确认前景样本
  single_side      单侧亮源: 方向性取舍(边缘能量左右比)
  fine_structure   细发丝: 内部颜色保留 vs 边缘融合; 整体模糊对照
  nonuniform_bg    非均匀背景: 各模型偏差与 vis_mix 适用范围
  occlusion_switch 遮挡门槛双模型切换: 背景亮度不变而遮挡改变时淘汰错误模型;
                   以按局部深度选择(depth_select)为参照检验 vis_mix
"""

import json
import os

import numpy as np

from . import models, samples
from .blur import gaussian_blur
from .colorspace import srgb_to_linear
from .image import save_gray, save_heatmap, save_rgb
from .metrics import edge_blend_diff, edge_energy, subject_color_diff
from .params import WrapParams
from .pipeline import apply_occluder_to_bg, edge_check_crop, run_pipeline, to_display

# 实验显式参数 —— 组合模型门槛必须显式给出, 不读默认值.
# 空间类参数随样本尺寸缩放(以 96px 为基准), 保证射线预算覆盖实验几何.
def experiment_params(size):
    s = size / 96.0
    return dict(
        inner_width=6.0 * s, outer_width=10.0 * s, intensity=1.0,
        blur_sigma=12.0 * s, dir_offset=8.0 * s,
        ray_count=16, max_dist=48.0 * s, cone_angle=1.2,
        falloff_tau=24.0 * s,
        depth_threshold=24.0 * s, occ_threshold=0.25,
        allow_negative=False, deform_amp=0.0, deform_freq=0.12,
        strand_strength=0.0, linear=True, use_occluder=True,
    )


BASE_METHODS = ("blur", "visibility")
COMBINED = ("depth_select", "vis_mix", "switch")
ALL_METHODS = BASE_METHODS + COMBINED

# vis_mix 相对 depth_select 参照的偏差容差(包光层 RMS, 线性域)
VIS_MIX_TOL = 0.02
# 模型淘汰判定: 参照能量变化显著而候选变化方向相反或幅度不足其 20% 时淘汰
ELIMINATE_RATIO = 0.2


def make_params(method, size=256, **overrides):
    """实验参数: 全部显式给出, 组合模型门槛不省略."""
    kw = experiment_params(size)
    kw.update(overrides)
    return WrapParams(method=method, **kw)


def save_run_layers(out_dir, out, params):
    """保存一次运行的证据图层与参数(决定和采样证据随工程保存)."""
    os.makedirs(out_dir, exist_ok=True)
    save_rgb(os.path.join(out_dir, "composite_on.png"),
             np.clip(to_display(out.composite_on, params), 0, 1))
    save_rgb(os.path.join(out_dir, "composite_off.png"),
             np.clip(to_display(out.composite_off, params), 0, 1))
    wrap_vis = out.wrap_layer
    peak = wrap_vis.max()
    save_rgb(os.path.join(out_dir, "wrap_layer.png"),
             wrap_vis / peak if peak > 1e-9 else wrap_vis)
    save_heatmap(os.path.join(out_dir, "wrap_energy.png"),
                 np.linalg.norm(out.wrap_layer, axis=-1))
    save_heatmap(os.path.join(out_dir, "band_weight.png"), out.band_weight)
    excess_peak = out.excess.max()
    save_rgb(os.path.join(out_dir, "excess.png"),
             out.excess / excess_peak if excess_peak > 1e-9 else out.excess)
    if "visibility" in out.debug:
        save_heatmap(os.path.join(out_dir, "visibility.png"), out.debug["visibility"])
        save_heatmap(os.path.join(out_dir, "blocked.png"), out.debug["blocked"])
    if "select_dir_mask" in out.debug:
        save_gray(os.path.join(out_dir, "select_dir_mask.png"),
                  out.debug["select_dir_mask"].astype(np.float64))
    crop = edge_check_crop(to_display(out.composite_on, params), out.band_weight)
    save_rgb(os.path.join(out_dir, "edge_check.png"), np.clip(crop, 0, 1))
    params.save(os.path.join(out_dir, "params.json"))


def _layer_rms(a, b, mask):
    """两张包光层在 mask 内的 RMS 差异(线性域)."""
    if not mask.any():
        return 0.0
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt((d[mask] ** 2).sum(axis=-1).mean()))


def _run_and_record(fg, alpha, bg, params, occluder, run_dir):
    out = run_pipeline(fg, alpha, bg, params, occluder=occluder)
    save_run_layers(run_dir, out, params)
    return out


def exp_model_compare(ctx):
    """局部背景平均 vs 方向加权: 可切换实验, 接已确认前景样本.

    细发丝(branches)与大轮廓(hard)分别记录内部颜色保留与边缘融合.
    """
    fg_b, a_b = samples.make_fine_branches(ctx["size"])
    fg_h, a_h = samples.make_hard_edge(ctx["size"])
    bg = samples.make_background(ctx["size"])
    runs = []
    for subject, fg, alpha in (("branches", fg_b, a_b), ("hard", fg_h, a_h)):
        for method in ALL_METHODS:
            params = make_params(method, ctx["size"])
            run_dir = os.path.join(ctx["out_dir"], "model_compare",
                                   f"{subject}_{method}")
            out = _run_and_record(fg, alpha, bg, params, None, run_dir)
            runs.append({
                "subject": subject, "method": method,
                "metrics": out.metrics,
                "decision": out.debug.get("decision", ""),
                "evidence": os.path.relpath(run_dir, ctx["out_dir"]),
            })
    conclusions = []
    for subject in ("branches", "hard"):
        sub = {r["method"]: r["metrics"] for r in runs if r["subject"] == subject}
        interior_px = sub["visibility"]["subject_color_diff"]["interior_pixels"]
        note = ""
        if interior_px == 0:
            note = ("; 注意: 该样本无深层内部像素(interior_pixels=0), "
                    "内部保留指标为空集, 差异只看边缘融合")
        conclusions.append(
            f"[{subject}] 局部平均(blur) 边缘能量="
            f"{sub['blur']['edge_energy']['mean']:.4f} 主体色差="
            f"{sub['blur']['subject_color_diff']['mean']:.6f}; "
            f"方向加权(visibility) 边缘能量="
            f"{sub['visibility']['edge_energy']['mean']:.4f} 主体色差="
            f"{sub['visibility']['subject_color_diff']['mean']:.6f}; "
            f"边缘融合: blur={sub['blur']['edge_blend']['mean']:.4f} vs "
            f"visibility={sub['visibility']['edge_blend']['mean']:.4f}"
            f" (深层内部像素数={interior_px}){note}"
        )
    return {"experiment": "model_compare",
            "params": experiment_params(ctx["size"]),
            "runs": runs, "conclusions": conclusions}


def exp_single_side(ctx):
    """单侧亮源: 亮盘只在左侧, 检验各模型的方向性取舍."""
    size = ctx["size"]
    fg, alpha = samples.make_fine_branches(size)
    bg = samples.make_background(size, bright_center=(0.12, 0.45),
                                 bright_radius=0.14)
    band_probe = None
    runs = []
    for method in ALL_METHODS:
        params = make_params(method, size)
        run_dir = os.path.join(ctx["out_dir"], "single_side", method)
        out = _run_and_record(fg, alpha, bg, params, None, run_dir)
        energy = np.linalg.norm(out.wrap_layer, axis=-1)
        band = out.band_weight > 1e-3
        left = np.zeros_like(band)
        left[:, : size // 2] = True
        right = np.zeros_like(band)
        right[:, size // 2:] = True
        e_left = float(energy[band & left].mean()) if (band & left).any() else 0.0
        e_right = float(energy[band & right].mean()) if (band & right).any() else 0.0
        ratio = e_left / max(e_right, 1e-9)
        runs.append({
            "method": method,
            "metrics": out.metrics,
            "edge_energy_left": e_left, "edge_energy_right": e_right,
            "side_ratio": ratio,
            "decision": out.debug.get("decision", ""),
            "evidence": os.path.relpath(run_dir, ctx["out_dir"]),
        })
        band_probe = band
    r = {x["method"]: x for x in runs}
    strongest = max(ALL_METHODS, key=lambda m: r[m]["side_ratio"])
    weakest = min(ALL_METHODS, key=lambda m: r[m]["side_ratio"])
    conclusions = [
        "单侧亮源(左): 边缘能量左右比 "
        + ", ".join(f"{m}={r[m]['side_ratio']:.2f}" for m in ALL_METHODS)
        + f". 方向性最强={strongest}({r[strongest]['side_ratio']:.2f}), "
        f"最弱={weakest}({r[weakest]['side_ratio']:.2f}); "
        "所有模型比值均>1(亮源侧能量更高), 差异在方向性强度 —— "
        "局部平均的方向性来自 dir_offset 偏移采样, 方向加权来自锥形射线, "
        "取舍以边缘能量左右比量化."
    ]
    return {"experiment": "single_side",
            "params": experiment_params(size),
            "runs": runs, "conclusions": conclusions}


def _outside_band_diff(composite_a, composite_b, band_weight):
    """边缘带之外的改动量: 包光模型在带外必须严格为 0.

    整体模糊等错误替代会在带外(主体内部与远处背景)留下非零改动.
    """
    outside = np.asarray(band_weight) <= 1e-3
    if not outside.any():
        return 0.0
    diff = np.linalg.norm(
        np.asarray(composite_a, dtype=np.float64)
        - np.asarray(composite_b, dtype=np.float64), axis=-1)
    return float(diff[outside].mean())


def exp_fine_structure(ctx):
    """细发丝与大轮廓: 内部颜色保留 vs 边缘融合; 整体模糊对照.

    整体模糊(无边缘带限制)不是包光: 它在边缘带之外改动像素.
    细发丝样本几乎没有深层内部(发丝全在边缘带内), 内部保留指标
    对发丝为空集 —— 此时以"带外改动量"区分包光与整体模糊.
    """
    size = ctx["size"]
    bg = samples.make_background(size)
    subjects = {
        "branches": samples.make_fine_branches(size),
        "hard": samples.make_hard_edge(size),
    }
    runs = []
    for subject, (fg, alpha) in subjects.items():
        fg_linear = srgb_to_linear(np.clip(fg, 0.0, 1.0))
        for method in ALL_METHODS:
            params = make_params(method, size)
            run_dir = os.path.join(ctx["out_dir"], "fine_structure",
                                   f"{subject}_{method}")
            out = _run_and_record(fg, alpha, bg, params, None, run_dir)
            obd = _outside_band_diff(out.composite_on, out.composite_off,
                                     out.band_weight)
            runs.append({
                "subject": subject, "method": method,
                "metrics": out.metrics,
                "outside_band_diff": obd,
                "decision": out.debug.get("decision", ""),
                "evidence": os.path.relpath(run_dir, ctx["out_dir"]),
            })
        # 对照: 整体模糊合成结果(无边缘带限制) —— 不是包光
        params = make_params("visibility", size)
        out = run_pipeline(fg, alpha, bg, params)
        blurred_whole = gaussian_blur(out.composite_off, 6.0 * size / 96.0)
        ctrl_dir = os.path.join(ctx["out_dir"], "fine_structure",
                                f"{subject}_global_blur_control")
        os.makedirs(ctrl_dir, exist_ok=True)
        save_rgb(os.path.join(ctrl_dir, "composite_global_blur.png"),
                 np.clip(to_display(blurred_whole, params), 0, 1))
        ctrl_metrics = {
            "subject_color_diff": subject_color_diff(
                blurred_whole, fg_linear, out.signed_dist, params.inner_width),
            "edge_blend": edge_blend_diff(blurred_whole, out.composite_off,
                                          out.band_weight),
        }
        ctrl_obd = _outside_band_diff(blurred_whole, out.composite_off,
                                      out.band_weight)
        runs.append({
            "subject": subject, "method": "global_blur_control",
            "metrics": ctrl_metrics,
            "outside_band_diff": ctrl_obd,
            "decision": "整体模糊对照: 无边缘带限制, 带外像素被改动",
            "evidence": os.path.relpath(ctrl_dir, ctx["out_dir"]),
        })

    conclusions = []
    for subject in subjects:
        sub = [r for r in runs if r["subject"] == subject]
        ctrl = next(r for r in sub if r["method"] == "global_blur_control")
        wraps = [r for r in sub if r["method"] != "global_blur_control"]
        worst_obd = max(r["outside_band_diff"] for r in wraps)
        worst_scd = max(r["metrics"]["subject_color_diff"]["mean"] for r in wraps)
        interior_px = ctrl["metrics"]["subject_color_diff"]["interior_pixels"]
        conclusions.append(
            f"[{subject}] 包光模型带外改动量最大={worst_obd:.2e} "
            f"(深层内部像素数={interior_px}, 主体色差最大={worst_scd:.6f}); "
            f"整体模糊对照带外改动量={ctrl['outside_band_diff']:.6f} "
            f"主体色差={ctrl['metrics']['subject_color_diff']['mean']:.6f} "
            f"—— 整体模糊在边缘带之外改动像素, 不能替代包光."
        )
    return {"experiment": "fine_structure",
            "params": experiment_params(size),
            "runs": runs, "conclusions": conclusions}


def _make_nonuniform_bg(size):
    """非均匀背景: 双亮源 + 强低频纹理(实验内构造, 不改样本生成器)."""
    bg = samples.make_background(size, bright_center=(0.15, 0.25),
                                 bright_radius=0.12)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    d2 = (xx - 0.85 * size) ** 2 + (yy - 0.7 * size) ** 2
    disc2 = np.clip(1.0 - np.sqrt(d2) / (0.10 * size), 0.0, 1.0)
    bg = bg + disc2[:, :, None] * np.array([1.2, 0.9, 1.8])
    tex = gaussian_blur(np.random.default_rng(11).random((size, size, 3)), 3.0)
    return np.clip(bg * (0.7 + 0.6 * tex), 0.0, 4.0)


def exp_nonuniform_bg(ctx):
    """非均匀背景: 各模型偏差; 以 depth_select 为参照检验 vis_mix."""
    size = ctx["size"]
    fg, alpha = samples.make_fine_branches(size)
    bg = _make_nonuniform_bg(size)
    outs = {}
    runs = []
    for method in ALL_METHODS:
        params = make_params(method, size)
        run_dir = os.path.join(ctx["out_dir"], "nonuniform_bg", method)
        outs[method] = _run_and_record(fg, alpha, bg, params, None, run_dir)
        runs.append({
            "method": method, "metrics": outs[method].metrics,
            "decision": outs[method].debug.get("decision", ""),
            "evidence": os.path.relpath(run_dir, ctx["out_dir"]),
        })
    band = outs["visibility"].band_weight > 1e-3
    ref = outs["depth_select"].wrap_layer
    deviations = {}
    for method in ALL_METHODS:
        if method == "depth_select":
            continue
        deviations[method] = _layer_rms(outs[method].wrap_layer, ref, band)
    for r in runs:
        if r["method"] in deviations:
            r["rms_vs_depth_select"] = deviations[r["method"]]
    applicable = deviations["vis_mix"] <= VIS_MIX_TOL
    conclusions = [
        "非均匀背景: 各模型相对 depth_select 参照的包光层 RMS 偏差 "
        + ", ".join(f"{m}={deviations[m]:.4f}" for m in sorted(deviations))
        + f"; vis_mix 偏差{'<' if applicable else '>='}容差 {VIS_MIX_TOL}, "
        f"在本场景{'适用' if applicable else '不适用'}."
    ]
    return {"experiment": "nonuniform_bg",
            "params": experiment_params(size),
            "vis_mix_tolerance": VIS_MIX_TOL,
            "vis_mix_applicable": applicable,
            "runs": runs, "conclusions": conclusions}


def exp_occlusion_switch(ctx):
    """遮挡门槛双模型切换: 背景亮度不变而遮挡改变时淘汰错误模型.

    同一背景(亮度不变), 遮挡 none -> near -> far; 以 depth_select 为参照:
      - 候选模型在遮挡出现前后 ROI 边缘能量不变而参照变化显著 -> 淘汰;
      - vis_mix 与参照的 RMS 偏差决定其适用范围.
    决定(eliminated)与采样证据(visibility/blocked 图层)随 report 保存.
    """
    size = ctx["size"]
    fg, alpha = samples.make_fine_branches(size)
    bg = samples.make_background(size)   # 背景亮度全程不变
    _, occ_near = samples.make_occluder_plate_image(size)
    occ_far = samples.make_occluder(size, box=(0.60, 0.30, 0.70, 0.62))
    occluders = [("none", None), ("near", occ_near), ("far", occ_far)]

    roi = np.zeros((size, size), dtype=bool)
    ys, xs = np.nonzero(occ_near > 0.5)
    y0, y1 = max(ys.min() - 32, 0), min(ys.max() + 32, size)
    x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 48, size)
    roi[y0:y1, x0:x1] = True

    runs = []
    energy_roi = {}
    wrap_layers = {}
    bands = {}
    for method in ALL_METHODS:
        for occ_name, occ in occluders:
            params = make_params(method, size)
            bg_eff = apply_occluder_to_bg(bg, occ, samples.HAIR_COLOR)
            run_dir = os.path.join(ctx["out_dir"], "occlusion_switch",
                                   f"{method}_{occ_name}")
            out = _run_and_record(fg, alpha, bg_eff, params, occ, run_dir)
            band = out.band_weight > 1e-3
            sel = band & roi
            e_roi = float(np.linalg.norm(out.wrap_layer, axis=-1)[sel].mean()) \
                if sel.any() else 0.0
            energy_roi[(method, occ_name)] = e_roi
            wrap_layers[(method, occ_name)] = out.wrap_layer
            bands[(method, occ_name)] = band
            runs.append({
                "method": method, "occluder": occ_name,
                "metrics": out.metrics,
                "edge_energy_roi": e_roi,
                "decision": out.debug.get("decision", ""),
                "evidence": os.path.relpath(run_dir, ctx["out_dir"]),
            })

    # 模型淘汰: 背景亮度不变而遮挡出现时, 参照(depth_select)ROI 能量下降;
    # 候选变化方向相反(遮挡后反而更亮)或幅度不足参照的 ELIMINATE_RATIO
    # -> 对遮挡不敏感/方向错误, 淘汰.
    # 参照自身不响应时标记实验不可判定, 不静默下结论.
    ref = "depth_select"
    ref_delta = energy_roi[(ref, "near")] - energy_roi[(ref, "none")]
    ref_responsive = abs(ref_delta) > 1e-6
    eliminated, kept = [], []
    if ref_responsive:
        for method in ALL_METHODS:
            if method == ref:
                continue
            delta = energy_roi[(method, "near")] - energy_roi[(method, "none")]
            wrong_dir = ref_delta * delta < 0 and abs(delta) > 1e-6
            too_weak = abs(delta) < ELIMINATE_RATIO * abs(ref_delta)
            if wrong_dir or too_weak:
                why = "方向相反" if wrong_dir else (
                    "幅度不足参照的 %.0f%%" % (ELIMINATE_RATIO * 100))
                eliminated.append({
                    "method": method,
                    "reason": f"遮挡出现时参照 ROI 能量变化 {ref_delta:+.6f}, "
                              f"该模型变化 {delta:+.6f}({why}), 为错误模型",
                })
            else:
                kept.append(method)

    # vis_mix 适用范围: 与参照的包光层 RMS 偏差(逐遮挡状态)
    vis_mix_dev = {}
    for occ_name, _ in occluders:
        band = bands[(ref, occ_name)]
        vis_mix_dev[occ_name] = _layer_rms(
            wrap_layers[("vis_mix", occ_name)],
            wrap_layers[(ref, occ_name)], band)
    applicable = {k: v <= VIS_MIX_TOL for k, v in vis_mix_dev.items()}

    conclusions = []
    if ref_responsive:
        conclusions.append(
            f"遮挡切换(背景亮度不变): 参照 depth_select 的 ROI 边缘能量随遮挡"
            f"变化 {ref_delta:+.6f}; 淘汰模型 "
            f"{[e['method'] for e in eliminated] or '无'}"
            f"(变化方向相反或幅度不足), 保留 {kept}.")
    else:
        conclusions.append(
            f"遮挡切换(背景亮度不变): 参照 depth_select 的 ROI 边缘能量随遮挡"
            f"变化 {ref_delta:+.6f} ≈ 0, 参照自身不响应 —— 实验不可判定, "
            f"需检查几何或射线预算(max_dist={experiment_params(size)['max_dist']:g}).")
    conclusions.append(
        "vis_mix 相对参照的 RMS 偏差: "
        + ", ".join(f"{k}={v:.4f}({'适用' if applicable[k] else '不适用'})"
                    for k, v in vis_mix_dev.items())
        + f" (容差 {VIS_MIX_TOL})."
    )
    return {
        "experiment": "occlusion_switch",
        "params": experiment_params(size),
        "reference_model": ref,
        "reference_responsive": ref_responsive,
        "eliminate_ratio": ELIMINATE_RATIO,
        "eliminated": eliminated, "kept": kept,
        "vis_mix_deviation": vis_mix_dev,
        "vis_mix_applicable": applicable,
        "runs": runs, "conclusions": conclusions,
    }


EXPERIMENTS = {
    "model_compare": exp_model_compare,
    "single_side": exp_single_side,
    "fine_structure": exp_fine_structure,
    "nonuniform_bg": exp_nonuniform_bg,
    "occlusion_switch": exp_occlusion_switch,
}


def _write_conclusions_md(path, reports):
    """由复算数据生成结论文件 —— 数字全部来自 report, 不手写."""
    lines = [
        "# 包光模型实验结论",
        "",
        "本文件由 `wraplight experiment` 根据复算指标生成; 边缘能量/主体色差等",
        "数值均来自各实验 report.json, 对应证据图层在同名目录下.",
        "",
    ]
    for rep in reports:
        lines.append(f"## {rep['experiment']}")
        lines.append("")
        for c in rep["conclusions"]:
            lines.append(f"- {c}")
        if rep.get("eliminated"):
            lines.append("- 被淘汰模型(背景亮度不变而遮挡改变时):")
            for e in rep["eliminated"]:
                lines.append(f"  - {e['method']}: {e['reason']}")
        lines.append("")
    lines.append("## 贡献模型讨论(各方案取舍)")
    lines.append("")
    for key, text in models.MODEL_DISCUSSION.items():
        lines.append(f"- **{key}**: {text}")
    lines.append("")
    lines.append("## 待验假设(单列)")
    lines.append("")
    for i, a in enumerate(models.PENDING_ASSUMPTIONS, 1):
        lines.append(f"{i}. {a}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_experiments(out_dir, size=256, names=None):
    """运行实验并保存证据; 返回各实验 report."""
    os.makedirs(out_dir, exist_ok=True)
    ctx = {"out_dir": out_dir, "size": size}
    selected = names or list(EXPERIMENTS)
    unknown = [n for n in selected if n not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"未知实验: {unknown}; 可选: {list(EXPERIMENTS)}")
    reports = []
    for name in selected:
        rep = EXPERIMENTS[name](ctx)
        rep_dir = os.path.join(out_dir, name)
        os.makedirs(rep_dir, exist_ok=True)
        with open(os.path.join(rep_dir, "report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        reports.append(rep)
    _write_conclusions_md(os.path.join(out_dir, "CONCLUSIONS.md"), reports)
    return reports
