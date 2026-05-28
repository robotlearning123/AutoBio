#!/usr/bin/env python3
"""Build dual MuJoCo + Isaac Sim demo report HTML."""
import json
import base64
from pathlib import Path

DEMO_DIR = Path(__file__).parent.parent / "demos"

TASKS = [
    ("insert_centrifuge_5430", "Insert into Centrifuge 5430", "Insertion", 7, "ur"),
    ("insert",                 "Insert Centrifuge Tube",      "Insertion",  6, "ur"),
    ("pipette",                "Pipette Liquid Transfer",     "Liquid",     6, "ur"),
    ("thermal_cycler_open",    "Open Thermal Cycler",         "Articulated",6, "ur"),
    ("thermal_cycler_close",   "Close Thermal Cycler",        "Articulated",6, "ur"),
    ("thermal_mixer",          "Thermal Mixer Operation",     "Articulated",6, "ur"),
    ("pickup",                 "Pickup Centrifuge Tube",      "Grasp",      3, "aloha"),
    ("screw_loose",            "Loosen Screw Cap",            "Rotary",     3, "aloha"),
    ("screw_tighten",          "Tighten Screw Cap",           "Rotary",     3, "aloha"),
    ("vortex_mixer",           "Vortex Mixer",                "Rotary",     4, "aloha"),
]


def build_report():
    cards = []
    mujoco_count = 0
    isaac_count = 0

    for task_id, title, category, score, robot in TASKS:
        mujoco_path = DEMO_DIR / f"{task_id}.mp4"
        isaac_path = DEMO_DIR / f"{task_id}_isaac.mp4"

        has_mujoco = mujoco_path.exists() and mujoco_path.stat().st_size > 10000
        has_isaac = isaac_path.exists() and isaac_path.stat().st_size > 10000

        if has_mujoco:
            mujoco_count += 1
        if has_isaac:
            isaac_count += 1

        mujoco_size = f"{mujoco_path.stat().st_size / 1024:.0f} KB" if has_mujoco else "N/A"
        isaac_size = f"{isaac_path.stat().st_size / 1024:.0f} KB" if has_isaac else "N/A"

        isaac_result_path = DEMO_DIR / f"{task_id}_isaac_result.json"
        isaac_info = ""
        isaac_badge = ""
        if isaac_result_path.exists():
            r = json.loads(isaac_result_path.read_text())
            hidden = r.get("hidden_meshes", 0)
            mean = r.get("first_mean", 0)
            diff = r.get("frame_diff", 0)
            anim = r.get("animated", False)
            colored = r.get("colored_meshes", 0)
            parts = [f"{colored}c"]
            if anim:
                parts.append(f"anim d={diff:.1f}")
            else:
                parts.append("static")
            isaac_info = " | ".join(parts)
            isaac_badge = "animated" if anim and diff > 2 else "colored"

        if has_mujoco:
            mujoco_html = f"""
            <div class="video-panel">
              <div class="video-label">MuJoCo 3.3 (EGL)</div>
              <video controls preload="metadata" class="demo-video">
                <source src="{task_id}.mp4" type="video/mp4">
              </video>
              <div class="video-meta">{mujoco_size}</div>
            </div>"""
        else:
            mujoco_html = """
            <div class="video-panel missing">
              <div class="video-label">MuJoCo 3.3 (EGL)</div>
              <div class="missing-text">Not available</div>
            </div>"""

        if has_isaac:
            isaac_html = f"""
            <div class="video-panel">
              <div class="video-label">Isaac Sim 5.1 (RTX)</div>
              <video controls preload="metadata" class="demo-video">
                <source src="{task_id}_isaac.mp4" type="video/mp4">
              </video>
              <div class="video-meta">{isaac_size} | {isaac_info}</div>
              {"<span class=\"badge isaac-badge\">" + isaac_badge + "</span>" if isaac_badge else ""}
            </div>"""
        else:
            isaac_html = """
            <div class="video-panel missing">
              <div class="video-label">Isaac Sim 5.1 (RTX)</div>
              <div class="missing-text">Not available</div>
            </div>"""

        score_color = "#34d399" if score >= 6 else ("#fbbf24" if score >= 5 else "#f87171")
        robot_badge = "UR5e + 2f85" if robot == "ur" else "ALOHA vx300s"
        render_note = "" if score >= 5 else '<div class="render-warn">Known issue: ALOHA arm body not rendered (UR works; root cause WIP). Gripper shown as end-effector preview.</div>'
        cards.append(f"""
    <div class="card" data-cat="{category}" data-score="{score}" data-robot="{robot}">
      <div class="card-header">
        <h3>{title}</h3>
        <div class="card-meta">
          <span class="badge">{category}</span>
          <span class="badge robot-{robot}">{robot_badge}</span>
          <span class="badge score" style="background:{score_color}22;color:{score_color};border:1px solid {score_color}55">Isaac {score}/10</span>
        </div>
      </div>
      <div class="card-body">
        {render_note}
        <div class="video-row">
          {mujoco_html}
          {isaac_html}
        </div>
      </div>
    </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoBio Dual Simulator Demo</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f1117; color: #e0e0e0; }}
.hero {{ background: linear-gradient(135deg, #1a1d2e 0%, #2d1b4e 50%, #1a2d3e 100%); padding: 40px 32px 24px; border-bottom: 1px solid #2a2d3a; }}
.hero h1 {{ font-size: 28px; font-weight: 700; color: #fff; margin-bottom: 6px; }}
.hero .subtitle {{ font-size: 14px; color: #8b8fa3; margin-bottom: 16px; }}
.stats {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.stat {{ background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 16px; min-width: 120px; }}
.stat .label {{ font-size: 11px; color: #6b7084; text-transform: uppercase; letter-spacing: 0.5px; }}
.stat .value {{ font-size: 22px; font-weight: 600; color: #a78bfa; margin-top: 2px; }}
.filters {{ padding: 16px 32px; display: flex; gap: 8px; border-bottom: 1px solid #1e2030; background: #12141e; position: sticky; top: 0; z-index: 10; }}
.filter-btn {{ background: rgba(255,255,255,0.05); border: 1px solid #2a2d3a; color: #8b8fa3; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; transition: all 0.15s; }}
.filter-btn:hover {{ background: rgba(167,139,250,0.1); color: #a78bfa; border-color: #a78bfa; }}
.filter-btn.active {{ background: rgba(167,139,250,0.15); color: #a78bfa; border-color: #a78bfa; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; padding: 24px 32px 48px; max-width: 1200px; margin: 0 auto; }}
.card {{ background: #181b28; border: 1px solid #2a2d3a; border-radius: 10px; overflow: hidden; transition: border-color 0.2s; }}
.card:hover {{ border-color: #4a4d5a; }}
.card-header {{ display: flex; justify-content: space-between; align-items: center; padding: 14px 18px 10px; }}
.card-header h3 {{ font-size: 16px; font-weight: 600; color: #fff; }}
.badge {{ background: rgba(167,139,250,0.15); color: #a78bfa; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
.isaac-badge {{ background: rgba(52,211,153,0.15); color: #34d399; margin-top: 4px; display: inline-block; }}
.card-body {{ padding: 0 18px 14px; }}
.video-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.video-panel {{ text-align: center; }}
.video-label {{ font-size: 12px; color: #8b8fa3; margin-bottom: 6px; font-weight: 500; }}
.demo-video {{ width: 100%; border-radius: 6px; background: #000; max-height: 280px; }}
.video-meta {{ font-size: 11px; color: #5a5e72; margin-top: 4px; }}
.video-panel.missing {{ display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 160px; background: rgba(255,255,255,0.02); border-radius: 6px; border: 1px dashed #2a2d3a; }}
.missing-text {{ color: #4a4d5a; font-size: 13px; margin-top: 8px; }}
.card-meta {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.robot-ur {{ background: rgba(96,165,250,0.15); color: #60a5fa; }}
.robot-aloha {{ background: rgba(244,114,182,0.15); color: #f472b6; }}
.render-warn {{ background: rgba(248,113,113,0.08); color: #fca5a5; padding: 8px 12px; border-radius: 6px; font-size: 12px; margin-bottom: 10px; border-left: 3px solid #f87171; }}
@media (max-width: 768px) {{ .video-row {{ grid-template-columns: 1fr; }} .grid {{ padding: 16px; gap: 16px; }} }}
</style>
</head>
<body>
<div class="hero">
  <h1>AutoBio Dual Simulator Demo</h1>
  <p class="subtitle">MuJoCo 3.3.0 (EGL) vs Isaac Sim 5.1 (RTX) | Newton USD | 1280x720 @ 24fps | Honest scoring (frame-inspected 2026-05-28)</p>
  <div class="stats">
    <div class="stat"><div class="label">MuJoCo Videos</div><div class="value">{mujoco_count}/10</div></div>
    <div class="stat"><div class="label">Isaac Videos</div><div class="value">{isaac_count}/10</div></div>
    <div class="stat"><div class="label">UR Full Arm</div><div class="value" style="color:#60a5fa">6/10</div></div>
    <div class="stat"><div class="label">ALOHA Arm WIP</div><div class="value" style="color:#f472b6">4/10</div></div>
  </div>
</div>
<div class="filters">
  <button class="filter-btn active" onclick="filter('all',this)">All ({len(TASKS)})</button>
  <button class="filter-btn" onclick="filter('Grasp',this)">Grasp</button>
  <button class="filter-btn" onclick="filter('Insertion',this)">Insertion</button>
  <button class="filter-btn" onclick="filter('Articulated',this)">Articulated</button>
  <button class="filter-btn" onclick="filter('Rotary',this)">Rotary</button>
  <button class="filter-btn" onclick="filter('Liquid',this)">Liquid</button>
</div>
<div class="grid">
{''.join(cards)}
</div>
<script>
function filter(cat, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.card').forEach(c => {{
    if (cat === 'all' || c.dataset.cat === cat) c.style.display = '';
    else c.style.display = 'none';
  }});
}}
</script>
</body>
</html>"""

    out = DEMO_DIR / "autobio_dual_report.html"
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"MuJoCo: {mujoco_count}/10, Isaac: {isaac_count}/10")


if __name__ == "__main__":
    build_report()
