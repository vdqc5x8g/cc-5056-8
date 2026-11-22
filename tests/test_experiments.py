"""仓库内实验的测试: 证据落盘、指标复算、旧样例运行到底."""

import json
import os

import numpy as np
import pytest

from wraplight import experiments, samples
from wraplight.params import WrapParams
from wraplight.pipeline import apply_occluder_to_bg, run_pipeline


def test_experiments_write_evidence(tmp_path):
    """实验运行到底: report.json 与结论文件落盘, 指标为代码复算."""
    out_dir = str(tmp_path / "experiments")
    reports = experiments.run_experiments(out_dir, size=96,
                                          names=["model_compare"])
    rep = reports[0]
    assert rep["experiment"] == "model_compare"
    report_path = os.path.join(out_dir, "model_compare", "report.json")
    assert os.path.isfile(report_path)
    with open(report_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["runs"], "实验必须包含运行记录"
    for run in saved["runs"]:
        m = run["metrics"]
        assert set(m) == {"edge_energy", "subject_color_diff", "edge_blend"}
        evidence = os.path.join(out_dir, run["evidence"])
        assert os.path.isfile(os.path.join(evidence, "params.json"))
        assert os.path.isfile(os.path.join(evidence, "composite_on.png"))
    assert os.path.isfile(os.path.join(out_dir, "CONCLUSIONS.md"))


def test_experiment_params_are_explicit():
    """包光模型实验不得以默认值遮错: 组合模型门槛显式给出."""
    for method in experiments.ALL_METHODS:
        p = experiments.make_params(method)
        if method == "depth_select":
            assert p.depth_threshold is not None
        if method == "switch":
            assert p.occ_threshold is not None


def test_occlusion_switch_report(tmp_path):
    """遮挡切换实验: 淘汰决定与 vis_mix 适用范围写入报告."""
    out_dir = str(tmp_path / "experiments")
    reports = experiments.run_experiments(out_dir, size=96,
                                          names=["occlusion_switch"])
    rep = reports[0]
    assert rep["reference_model"] == "depth_select"
    methods = {e["method"] for e in rep["eliminated"]}
    assert "blur" in methods, "背景亮度不变而遮挡改变时, 局部平均应被淘汰"
    assert set(rep["vis_mix_deviation"]) == {"none", "near", "far"}
    assert set(rep["vis_mix_applicable"]) == {"none", "near", "far"}
    # 采样证据: 组合模型运行目录含可见性/遮挡图
    vis_dir = os.path.join(out_dir, "occlusion_switch", "visibility_near")
    assert os.path.isfile(os.path.join(vis_dir, "visibility.png"))
    assert os.path.isfile(os.path.join(vis_dir, "blocked.png"))


def test_unknown_experiment_rejected(tmp_path):
    with pytest.raises(ValueError, match="未知实验"):
        experiments.run_experiments(str(tmp_path), size=64, names=["nope"])


def test_old_samples_run_to_completion(tmp_path):
    """旧样例(已确认前景)用全部方法运行到底, 主体内部不被改变."""
    fg, alpha = samples.make_fine_branches(96)
    bg = samples.make_background(96)
    _, occ = samples.make_occluder_plate_image(96)
    bg_eff = apply_occluder_to_bg(bg, occ, samples.HAIR_COLOR)
    for method in experiments.ALL_METHODS:
        params = experiments.make_params(method)
        out = run_pipeline(fg, alpha, bg_eff, params, occluder=occ)
        assert out.composite_on.shape == fg.shape
        scd = out.metrics["subject_color_diff"]
        assert scd["mean"] < 1e-9, f"{method} 改变了主体内部"
        assert out.metrics["edge_energy"]["band_pixels"] > 0


def test_conclusions_contain_recomputed_numbers(tmp_path):
    """结论文件中的数字来自复算指标(与 report.json 一致)."""
    out_dir = str(tmp_path / "experiments")
    experiments.run_experiments(out_dir, size=96, names=["single_side"])
    with open(os.path.join(out_dir, "single_side", "report.json"),
              encoding="utf-8") as f:
        rep = json.load(f)
    with open(os.path.join(out_dir, "CONCLUSIONS.md"), encoding="utf-8") as f:
        text = f.read()
    vis = next(r for r in rep["runs"] if r["method"] == "visibility")
    assert f"{vis['side_ratio']:.2f}" in text
