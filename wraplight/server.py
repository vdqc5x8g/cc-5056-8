import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from . import limitations, models, samples
from .image import load_alpha, load_mask, load_rgb, save_gray, save_heatmap, save_rgb
from .params import WrapParams
from .pipeline import (
    apply_occluder_to_bg,
    edge_check_crop,
    run_pipeline,
    to_display,
)

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>前景边缘包光合成工具</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 0; background:#1c1f24; color:#dde;
        display:flex; min-height:100vh; }
 #side { width: 320px; padding: 14px; background:#262a31; overflow-y:auto;
         max-height:100vh; box-sizing:border-box; }
 #main { flex:1; padding:14px; }
 label { display:block; font-size:12px; margin-top:8px; }
 input[type=range]{ width:100%; }
 select,button { width:100%; margin-top:6px; padding:6px; }
 .val { float:right; color:#9ab; }
 #tabs button { width:auto; margin:2px; padding:6px 10px; }
 #tabs button.active { background:#4a90d9; color:#fff; }
 img.view { max-width:100%; image-rendering:pixelated; background:#111;
            border:1px solid #444; }
 .note { font-size:12px; color:#9ab; }
 #lims { font-size:12px; line-height:1.5; }
 #lims li { margin-bottom:4px; }
 h3 { margin:10px 0 4px; font-size:14px; }
 #report { font-size:12px; white-space:pre-wrap; background:#111; padding:8px;
           border:1px solid #444; }
 #report.error { border-color:#c44; color:#fbb; }
 .slider-off { opacity:0.45; }
 .thresh { display:flex; align-items:center; gap:6px; }
 .thresh input[type=checkbox]{ width:auto; }
</style>
</head>
<body>
<div id="side">
  <h3>输入</h3>
  <label>前景样本</label>
  <select id="subject">
    <option value="branches">细枝(发丝级)</option>
    <option value="hard">硬边</option>
  </select>
  <label>挡板(与发丝同色)</label>
  <select id="occluder">
    <option value="none">无</option>
    <option value="near">亮源与主体之间</option>
    <option value="far">远离亮源</option>
  </select>
  <h3>包光参数</h3>
  <label>估算方法(贡献模型)</label>
  <select id="method"></select>
  <div id="sliders"></div>
  <label><input type="checkbox" id="linear" checked> 线性光度(关闭仅作对照)</label>
  <label><input type="checkbox" id="allow_negative"> 允许负超出量</label>
  <button id="run">运行</button>
  <button id="save">保存参数与结果</button>
  <div id="report"></div>
  <h3>近似与伪影(可核对)</h3>
  <div id="lims"></div>
</div>
<div id="main">
  <div id="tabs"></div>
  <div id="view"></div>
  <p class="note">包光层为能量伪彩; 边缘回查为替换背景后真实边缘的放大裁剪;
  轮廓对照为 包光关|包光开 并排裁剪。本工具是对环境包裹光的近似,
  并非完整真实光照模拟。</p>
</div>
<script>
/* 控件由边缘包光内核驱动: /api/controls 返回方法列表与滑条规格. */
const VIEWS = [["composite_on","合成·包光开"],["composite_off","合成·包光关"],
 ["edge_compare","轮廓对照(关|开)"],["wrap_energy","包光层"],["band_weight","边缘带"],
 ["visibility","可见性"],["blocked","遮挡比例"],["excess","超出量"],
 ["edge_check","边缘回查"]];
let lastRun = null;
let CONTROLS = null;

async function loadControls() {
  CONTROLS = await fetch("/api/controls").then(r=>r.json());
  const methodSel = document.getElementById("method");
  for (const m of CONTROLS.methods) {
    const o = document.createElement("option");
    o.value = m.id; o.textContent = m.label;
    methodSel.appendChild(o);
  }
  const sliders = document.getElementById("sliders");
  for (const s of CONTROLS.sliders) {
    const div = document.createElement("div");
    div.id = "box_" + s.key;
    if (s.default === null) {
      /* 门槛类控件无默认值: 需显式勾选启用, 否则内核报错并反馈 */
      div.innerHTML = `<div class="thresh">
        <input type="checkbox" id="en_${s.key}">
        <label for="en_${s.key}" style="margin:0">${s.label}(需显式设置)
        <span class="val" id="v_${s.key}"></span></label></div>
        <input type="range" id="${s.key}" min="${s.min}" max="${s.max}"
         step="${s.step}" disabled>`;
    } else {
      div.innerHTML = `<label>${s.label}<span class="val" id="v_${s.key}"></span></label>
       <input type="range" id="${s.key}" min="${s.min}" max="${s.max}" step="${s.step}">`;
    }
    sliders.appendChild(div);
    const el = document.getElementById(s.key);
    el.value = (s.default === null) ? s.min : s.default;
    const show = () => document.getElementById("v_"+s.key).textContent = el.value;
    el.oninput = show; show();
    if (s.default === null) {
      const en = document.getElementById("en_"+s.key);
      en.onchange = () => { el.disabled = !en.checked;
        document.getElementById("box_"+s.key).classList.toggle("slider-off", !en.checked); };
      en.onchange();
    }
  }
  methodSel.onchange = applyMethodFilter;
  applyMethodFilter();
}

function applyMethodFilter() {
  /* 按内核规格只显示当前方法相关的控件 */
  const m = document.getElementById("method").value;
  for (const s of CONTROLS.sliders) {
    const box = document.getElementById("box_" + s.key);
    const relevant = !s.methods || s.methods.includes(m);
    box.style.display = relevant ? "" : "none";
  }
}

const tabs = document.getElementById("tabs");
for (const [key,label] of VIEWS) {
  const b = document.createElement("button");
  b.textContent = label; b.dataset.view = key;
  b.onclick = () => showView(key);
  tabs.appendChild(b);
}
function showView(key) {
  for (const b of tabs.children) b.classList.toggle("active", b.dataset.view===key);
  const view = document.getElementById("view");
  view.innerHTML = "";
  if (!lastRun || !lastRun.views[key]) { view.textContent = "请先运行"; return; }
  const img = document.createElement("img");
  img.className = "view"; img.src = lastRun.views[key] + "?t=" + Date.now();
  view.appendChild(img);
}
function gatherParams() {
  const p = {};
  for (const s of CONTROLS.sliders) {
    const el = document.getElementById(s.key);
    if (s.default === null) {
      const en = document.getElementById("en_"+s.key);
      p[s.key] = en.checked ? parseFloat(el.value) : null;
    } else {
      p[s.key] = parseFloat(el.value);
    }
  }
  p.ray_count = Math.round(p.ray_count);
  p.method = document.getElementById("method").value;
  p.linear = document.getElementById("linear").checked;
  p.allow_negative = document.getElementById("allow_negative").checked;
  p.subject = document.getElementById("subject").value;
  p.occluder = document.getElementById("occluder").value;
  return p;
}
async function post(url, body) {
  const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"},
   body: JSON.stringify(body)});
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
document.getElementById("run").onclick = async () => {
  const rep = document.getElementById("report");
  rep.className = ""; rep.textContent = "计算中…";
  try {
    lastRun = await post("/api/run", gatherParams());
    rep.textContent = JSON.stringify(lastRun.report, null, 2);
    showView("composite_on");
  } catch (e) {
    /* 内核失败反馈回边缘包光界面 */
    rep.className = "error";
    rep.textContent = "包光内核错误: " + e.message;
  }
};
document.getElementById("save").onclick = async () => {
  if (!lastRun) { alert("请先运行"); return; }
  const r = await post("/api/save", {run: lastRun.run_id, params: gatherParams()});
  document.getElementById("report").textContent = "已保存: " + r.path;
};
fetch("/api/limitations").then(r=>r.json()).then(d => {
  const el = document.getElementById("lims");
  el.innerHTML = "<b>支持的近似</b><ol>" +
    d.approximations.map(t=>`<li>${t}</li>`).join("") +
    "</ol><b>易产生伪影的位置</b><ol>" +
    d.artifacts.map(t=>`<li>${t}</li>`).join("") +
    "</ol><b>待验假设</b><ol>" +
    d.assumptions.map(t=>`<li>${t}</li>`).join("") + "</ol>";
});
loadControls().then(() => showView("composite_on"));
</script>
</body>
</html>
"""


class _State:
    def __init__(self, samples_dir, outputs_dir):
        self.samples_dir = samples_dir
        self.outputs_dir = outputs_dir
        self.counter = 0
        os.makedirs(outputs_dir, exist_ok=True)
        if not os.path.exists(os.path.join(samples_dir, "background.png")):
            samples.generate_samples(samples_dir)


def _load_inputs(state, subject, occluder):
    s = state.samples_dir
    fg = load_rgb(os.path.join(s, f"fg_{subject}.png"))
    alpha = load_alpha(os.path.join(s, f"alpha_{subject}.png"))
    bg = load_rgb(os.path.join(s, "background.png"))
    occ = None
    if occluder != "none":
        occ = load_mask(os.path.join(s, f"occluder_{occluder}.png"))
    return fg, alpha, bg, occ


def _edge_report(out):
    energy = np.linalg.norm(out.wrap_layer, axis=-1)
    band = out.band_weight > 1e-3
    inner = out.params.inner_width if out.params else 6.0
    inside = out.signed_dist > inner + 1.0
    report = {
        "band_pixels": int(band.sum()),
        "wrap_energy_mean_in_band": round(float(energy[band].mean()), 6) if band.any() else 0.0,
        "wrap_energy_max_in_band": round(float(energy[band].max()), 6) if band.any() else 0.0,
        "wrap_energy_beyond_band_inside": round(float(energy[inside].max()), 6) if inside.any() else 0.0,
    }
    if out.metrics:
        report["metrics"] = out.metrics
    if out.debug.get("decision"):
        report["decision"] = out.debug["decision"]
    return report


def _edge_compare_strip(out, params):
    """轮廓对照: 边缘带裁剪 包光关|包光开 并排, 便于对照轮廓亮边与合成结果."""
    off = edge_check_crop(to_display(out.composite_off, params), out.band_weight)
    on = edge_check_crop(to_display(out.composite_on, params), out.band_weight)
    h = max(off.shape[0], on.shape[0])
    w = max(off.shape[1], on.shape[1])
    def pad(img):
        canvas = np.zeros((h, w, 3), dtype=np.float64)
        canvas[: img.shape[0], : img.shape[1]] = img
        return canvas
    off, on = pad(off), pad(on)
    divider = np.ones((h, 2, 3), dtype=np.float64) * 0.9
    return np.concatenate([off, divider, on], axis=1)


def _write_views(run_dir, out, params):
    save_rgb(os.path.join(run_dir, "composite_on.png"),
             np.clip(to_display(out.composite_on, params), 0, 1))
    save_rgb(os.path.join(run_dir, "composite_off.png"),
             np.clip(to_display(out.composite_off, params), 0, 1))
    save_heatmap(os.path.join(run_dir, "wrap_energy.png"),
                 np.linalg.norm(out.wrap_layer, axis=-1))
    save_heatmap(os.path.join(run_dir, "band_weight.png"), out.band_weight)
    peak = out.excess.max()
    save_rgb(os.path.join(run_dir, "excess.png"),
             out.excess / peak if peak > 1e-9 else out.excess)
    if "visibility" in out.debug:
        save_heatmap(os.path.join(run_dir, "visibility.png"), out.debug["visibility"])
        save_heatmap(os.path.join(run_dir, "blocked.png"), out.debug["blocked"])
    else:
        save_gray(os.path.join(run_dir, "visibility.png"), np.zeros_like(out.band_weight))
        save_gray(os.path.join(run_dir, "blocked.png"), np.zeros_like(out.band_weight))
    crop = edge_check_crop(to_display(out.composite_on, params), out.band_weight)
    save_rgb(os.path.join(run_dir, "edge_check.png"), np.clip(crop, 0, 1))
    save_rgb(os.path.join(run_dir, "edge_compare.png"),
             np.clip(_edge_compare_strip(out, params), 0, 1))


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json_body(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif path == "/api/controls":
                self._send(200, json.dumps(models.kernel_controls(),
                                           ensure_ascii=False))
            elif path == "/api/limitations":
                self._send(200, json.dumps({
                    "approximations": limitations.APPROXIMATIONS,
                    "artifacts": limitations.ARTIFACT_PRONE,
                    "assumptions": models.PENDING_ASSUMPTIONS,
                    "discussion": models.MODEL_DISCUSSION,
                }, ensure_ascii=False))
            elif path.startswith("/outputs/") or path.startswith("/samples/"):
                self._serve_file(path.lstrip("/"))
            else:
                self._send(404, "{}")

        def _serve_file(self, rel):
            root = os.path.abspath(".")
            full = os.path.abspath(rel)
            if not full.startswith(root) or not os.path.isfile(full):
                self._send(404, "{}")
                return
            with open(full, "rb") as f:
                self._send(200, f.read(), "image/png")

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._json_body()
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}))
                return
            if path == "/api/run":
                self._api_run(body)
            elif path == "/api/save":
                self._api_save(body)
            else:
                self._send(404, "{}")

        def _api_run(self, body):
            subject = body.pop("subject", "branches")
            occluder = body.pop("occluder", "none")
            params = WrapParams.from_dict(body)
            fg, alpha, bg, occ = _load_inputs(state, subject, occluder)
            bg = apply_occluder_to_bg(bg, occ, samples.HAIR_COLOR)
            try:
                out = run_pipeline(fg, alpha, bg, params, occluder=occ)
            except ValueError as e:
                # 内核失败反馈回边缘包光界面(如门槛未显式设置)
                self._send(400, json.dumps({"error": str(e)}, ensure_ascii=False))
                return
            state.counter += 1
            run_id = f"ui_run_{state.counter}"
            run_dir = os.path.join(state.outputs_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            _write_views(run_dir, out, params)
            params.save(os.path.join(run_dir, "params.json"))
            views = {}
            for name in ("composite_on", "composite_off", "edge_compare",
                         "wrap_energy", "band_weight", "visibility", "blocked",
                         "excess", "edge_check"):
                views[name] = f"/outputs/{run_id}/{name}.png"
            self._send(200, json.dumps({
                "run_id": run_id, "views": views, "report": _edge_report(out),
            }, ensure_ascii=False))

        def _api_save(self, body):
            run_id = body.get("run", "")
            run_dir = os.path.join(state.outputs_dir, os.path.basename(run_id))
            if not os.path.isdir(run_dir):
                self._send(400, json.dumps({"error": "run 不存在"}))
                return
            dest = os.path.join(state.outputs_dir,
                                "saved_" + time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(dest, exist_ok=True)
            for name in os.listdir(run_dir):
                if name.endswith(".png"):
                    with open(os.path.join(run_dir, name), "rb") as fsrc, \
                            open(os.path.join(dest, name), "wb") as fdst:
                        fdst.write(fsrc.read())
            WrapParams.from_dict(body.get("params", {})).save(
                os.path.join(dest, "params.json"))
            self._send(200, json.dumps({"path": dest}, ensure_ascii=False))

    return Handler


def serve(port, samples_dir, outputs_dir):
    state = _State(samples_dir, outputs_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    print(f"Web 界面: http://127.0.0.1:{port}/")
    server.serve_forever()
