import argparse
import json
import os
import sys

import numpy as np

from . import experiments, limitations, models, samples
from .colorspace import linear_to_srgb
from .image import load_alpha, load_mask, load_rgb, save_gray, save_heatmap, save_rgb
from .params import WrapParams
from .pipeline import apply_occluder_to_bg, edge_check_crop, run_pipeline, to_display


def _apply_overrides(params, overrides):
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not hasattr(params, key):
            raise SystemExit(f"未知参数: {key}")
        cur = getattr(params, key)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        else:
            val = raw
        setattr(params, key, val)
    return params


def _save_layers(out_dir, out, params):
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
    crop = edge_check_crop(to_display(out.composite_on, params), out.band_weight)
    save_rgb(os.path.join(out_dir, "edge_check.png"), np.clip(crop, 0, 1))
    params.save(os.path.join(out_dir, "params.json"))


def cmd_sample(args):
    samples.generate_samples(args.out, size=args.size)
    print(f"样本已生成到 {args.out}")


def cmd_run(args):
    params = WrapParams.load(args.params) if args.params else WrapParams()
    params = _apply_overrides(params, args.set)
    fg = load_rgb(args.fg)
    alpha = load_alpha(args.alpha)
    bg = load_rgb(args.bg)
    occ = load_mask(args.occluder) if args.occluder else None
    if occ is not None:
        plate = tuple(float(v) for v in args.plate_color.split(","))
        bg = apply_occluder_to_bg(bg, occ, plate)
    try:
        out = run_pipeline(fg, alpha, bg, params, occluder=occ)
    except ValueError as e:
        raise SystemExit(f"包光内核错误: {e}")
    _save_layers(args.out, out, params)
    _print_edge_report(out)
    print(f"结果已写入 {args.out}")


def _edge_energy(out, mask):
    energy = np.linalg.norm(out.wrap_layer, axis=-1)
    m = mask > 0.5
    if not m.any():
        return 0.0
    return float(energy[m].mean())


def _print_edge_report(out):
    energy = np.linalg.norm(out.wrap_layer, axis=-1)
    band = out.band_weight > 1e-3
    inner = out.params.inner_width if out.params else 6.0
    inside = out.signed_dist > inner + 1.0
    report = {
        "band_pixels": int(band.sum()),
        "wrap_energy_mean_in_band": float(energy[band].mean()) if band.any() else 0.0,
        "wrap_energy_max_in_band": float(energy[band].max()) if band.any() else 0.0,
        "wrap_energy_beyond_band_inside": float(energy[inside].max()) if inside.any() else 0.0,
    }
    if out.metrics:
        report["metrics"] = out.metrics
    if out.debug.get("decision"):
        report["decision"] = out.debug["decision"]
    print("边缘回查:", json.dumps(report, ensure_ascii=False, indent=2))


def cmd_compare(args):
    """挡板对照实验.

    亮源附近的挡板(与发丝同色)放在两个位置; 分别用 blur / falloff /
    visibility 三种估算, 输出遮后边缘颜色对照与包光能量统计 ——
    亮物被挡板遮住后, visibility 法应停止包光, blur 法不会.
    """
    size = args.size
    fg, alpha = samples.make_fine_branches(size)
    bg = samples.make_background(size)
    _, occ_near = samples.make_occluder_plate_image(size)
    occ_far = samples.make_occluder(size, box=(0.60, 0.30, 0.70, 0.62))

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for occ_name, occ in [("none", None), ("near", occ_near), ("far", occ_far)]:
        bg_eff = apply_occluder_to_bg(bg, occ, samples.HAIR_COLOR)
        roi = None
        if occ is not None:
            ys, xs = np.nonzero(occ > 0.5)
            roi = np.zeros_like(occ, dtype=bool)
            y0, y1 = max(ys.min() - 32, 0), min(ys.max() + 32, size)
            x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 48, size)
            roi[y0:y1, x0:x1] = True
        for method in ("blur", "falloff", "visibility"):
            params = WrapParams(method=method, use_occluder=True)
            out = run_pipeline(fg, alpha, bg_eff, params, occluder=occ)
            tag = f"{method}_{occ_name}"
            _save_layers(os.path.join(args.out, tag), out, params)
            band = out.band_weight > 1e-3
            energy = np.linalg.norm(out.wrap_layer, axis=-1)
            row = {
                "method": method,
                "occluder": occ_name,
                "wrap_energy_mean": float(energy[band].mean()) if band.any() else 0.0,
            }
            if roi is not None:
                sel = band & roi
                row["wrap_energy_roi"] = float(energy[sel].mean()) if sel.any() else 0.0
            rows.append(row)
    report_path = os.path.join(args.out, "compare_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    for r in rows:
        roi_str = (f" roi={r['wrap_energy_roi']:.6f}" if "wrap_energy_roi" in r
                   else "")
        print(f"{r['method']:<11} occluder={r['occluder']:<5} "
              f"wrap_energy_mean={r['wrap_energy_mean']:.6f}{roi_str}")
    print(f"对照结果已写入 {args.out}")


def cmd_experiment(args):
    """仓库内包光模型实验: 结论与采样证据写入 outputs 目录."""
    names = args.name if args.name else None
    reports = experiments.run_experiments(args.out, size=args.size, names=names)
    for rep in reports:
        print(f"[{rep['experiment']}]")
        for c in rep["conclusions"]:
            print(f"  {c}")
        if rep.get("eliminated"):
            for e in rep["eliminated"]:
                print(f"  淘汰: {e['method']} —— {e['reason']}")
    print(f"实验证据与结论已写入 {args.out} (CONCLUSIONS.md)")


def cmd_limitations(_args):
    print("支持的近似:")
    for i, t in enumerate(limitations.APPROXIMATIONS, 1):
        print(f"  {i}. {t}")
    print("\n易产生伪影的位置:")
    for i, t in enumerate(limitations.ARTIFACT_PRONE, 1):
        print(f"  {i}. {t}")
    print("\n贡献模型讨论的待验假设(单列):")
    for i, t in enumerate(models.PENDING_ASSUMPTIONS, 1):
        print(f"  {i}. {t}")


def cmd_serve(args):
    from .server import serve
    serve(args.port, args.samples, args.outputs)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wraplight", description="前景边缘包光合成工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="生成样本图像")
    p.add_argument("--out", default="samples")
    p.add_argument("--size", type=int, default=256)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("run", help="运行包光合成")
    p.add_argument("--fg", required=True)
    p.add_argument("--alpha", required=True)
    p.add_argument("--bg", required=True)
    p.add_argument("--occluder", default=None)
    p.add_argument("--plate-color", default=",".join(str(v) for v in samples.HAIR_COLOR),
                   help="挡板颜色 r,g,b (sRGB 0..1)")
    p.add_argument("--params", default=None)
    p.add_argument("--set", action="append", default=[], help="覆盖参数 key=value")
    p.add_argument("--out", default="outputs/run")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="挡板对照实验(blur/falloff/visibility)")
    p.add_argument("--out", default="outputs/compare")
    p.add_argument("--size", type=int, default=256)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("experiment", help="包光模型实验(结论与证据随工程保存)")
    p.add_argument("--name", action="append", default=[],
                   help="实验名(可多次); 缺省运行全部: " +
                        ",".join(experiments.EXPERIMENTS))
    p.add_argument("--out", default="outputs/experiments")
    p.add_argument("--size", type=int, default=256)
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("limitations", help="列出支持的近似与易伪影位置")
    p.set_defaults(func=cmd_limitations)

    p = sub.add_parser("serve", help="启动 Web 界面")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--samples", default="samples")
    p.add_argument("--outputs", default="outputs")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
