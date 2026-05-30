"""
Generate confusion matrix + full HTML report from test_report.json
"""

import json, pathlib
from collections import defaultdict

CLASSES = ["bleeding", "glioma", "ischemia", "meningioma", "normal", "pituitary"]
ROOT    = pathlib.Path(__file__).parent
REPORT  = ROOT / "test_report.json"
OUT     = ROOT / "test_report.html"

# ── Build confusion matrix ──────────────────────────────────────────────────
data = json.loads(REPORT.read_text(encoding="utf-8"))

# matrix[actual][predicted] = count
matrix = {c: defaultdict(int) for c in CLASSES}

for actual, stats in data["per_class"].items():
    matrix[actual][actual] = stats["correct"]
    for w in stats["wrong"]:
        pred = w["predicted"].strip().lower()
        if pred in CLASSES:
            matrix[actual][pred] += 1

# Per-class metrics
per_class = {}
for cls in CLASSES:
    tp  = matrix[cls][cls]
    tot = data["per_class"][cls]["total"]
    fp  = sum(matrix[c][cls] for c in CLASSES if c != cls)
    fn  = tot - tp
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0
    rec  = tp / tot * 100 if tot else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    per_class[cls] = {"tp": tp, "total": tot, "fp": fp, "fn": fn,
                      "precision": prec, "recall": rec, "f1": f1}

overall = data["overall_accuracy"]

# ── HTML ────────────────────────────────────────────────────────────────────
def cell_color(val, row_total, is_diag):
    pct = val / row_total if row_total else 0
    if is_diag:
        r = int(255 - pct * 120)
        g = int(180 + pct * 75)
        b = int(180 + pct * 75)
        return f"rgb({r},{g},{b})"
    else:
        if val == 0:
            return "#f8fffe"
        intensity = min(pct * 8, 1.0)
        r = int(255)
        g = int(235 - intensity * 100)
        b = int(235 - intensity * 100)
        return f"rgb({r},{g},{b})"


matrix_rows = ""
for actual in CLASSES:
    row_total = data["per_class"][actual]["total"]
    cells = ""
    for predicted in CLASSES:
        val = matrix[actual][predicted]
        is_diag = actual == predicted
        bg = cell_color(val, row_total, is_diag)
        pct = val / row_total * 100 if row_total else 0
        bold = "font-weight:700;" if is_diag else ""
        sub = f'<div class="cell-pct">{pct:.0f}%</div>' if val > 0 else ""
        cells += f'<td style="background:{bg};{bold}">{val}{sub}</td>'
    matrix_rows += f"<tr><th class='row-label'>{actual}</th>{cells}</tr>"

bar_rows = ""
for cls in CLASSES:
    m = per_class[cls]
    acc = m["tp"] / m["total"] * 100 if m["total"] else 0
    fill = "#0EA5A0" if acc >= 95 else "#F59E0B" if acc >= 85 else "#EF4444"
    bar_rows += f"""
    <tr>
      <td class="cls-name">{cls}</td>
      <td class="bar-cell">
        <div class="bar-bg">
          <div class="bar-fill" style="width:{acc:.1f}%;background:{fill}"></div>
        </div>
      </td>
      <td class="num">{acc:.1f}%</td>
      <td class="num">{m['tp']}/{m['total']}</td>
      <td class="num">{m['precision']:.1f}%</td>
      <td class="num">{m['recall']:.1f}%</td>
      <td class="num">{m['f1']:.1f}</td>
    </tr>"""

col_headers = "".join(f"<th class='col-label'>{c}</th>" for c in CLASSES)

wrong_tables = ""
for cls in CLASSES:
    wrongs = data["per_class"][cls]["wrong"]
    if not wrongs:
        continue
    rows = ""
    for w in wrongs:
        pred = w["predicted"] or "(no response)"
        conf = w["confidence"]
        color = "#EF4444" if conf > 70 else "#F59E0B"
        rows += f"""<tr>
          <td class="fname">{w['file']}</td>
          <td><span class="pred-tag">{pred}</span></td>
          <td style="color:{color};font-weight:600">{conf}%</td>
        </tr>"""
    wrong_tables += f"""
    <div class="wrong-section">
      <h4><span class="cls-dot" style="background:var(--{cls})"></span>
        {cls} — {len(wrongs)} misclassified / {data['per_class'][cls]['total']} total
      </h4>
      <table class="wrong-table">
        <thead><tr><th>File</th><th>Predicted as</th><th>Confidence</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BrainScope AI — Test Report</title>
<style>
  :root {{
    --teal:#0EA5A0; --teal-dark:#0b8a85; --teal-pale:#E6F7F7;
    --dark:#0D1F2D; --mid:#3D5A6B; --light:#8AA8B8; --border:#E2EEF0;
    --bg:#F8FFFE;
    --bleeding:#EF4444; --glioma:#F97316; --ischemia:#8B5CF6;
    --meningioma:#0EA5A0; --normal:#10B981; --pituitary:#F59E0B;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',sans-serif; background:var(--bg); color:var(--dark); }}

  header {{
    background:linear-gradient(135deg,var(--dark),#152535);
    color:white; padding:32px 5%; display:flex; align-items:center; gap:16px;
  }}
  .logo-icon {{
    width:48px; height:48px; border-radius:12px;
    background:linear-gradient(135deg,var(--teal),#5ECFCA);
    display:flex; align-items:center; justify-content:center;
  }}
  header h1 {{ font-size:1.6rem; font-weight:800; }}
  header p  {{ color:rgba(255,255,255,.55); font-size:.9rem; margin-top:4px; }}

  main {{ max-width:1100px; margin:0 auto; padding:40px 24px; }}

  /* ── stat cards ── */
  .stat-cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:40px; }}
  .stat-card {{
    background:white; border:1px solid var(--border); border-radius:14px;
    padding:20px 24px;
  }}
  .stat-card .val {{ font-size:2rem; font-weight:800; color:var(--teal); }}
  .stat-card .lbl {{ font-size:.82rem; color:var(--light); margin-top:4px; }}

  /* ── section ── */
  .section {{ background:white; border:1px solid var(--border); border-radius:16px; padding:28px; margin-bottom:28px; }}
  .section h2 {{ font-size:1.15rem; font-weight:700; margin-bottom:20px; display:flex; align-items:center; gap:10px; }}
  .section h2::before {{ content:''; display:block; width:4px; height:20px; background:var(--teal); border-radius:2px; }}

  /* ── confusion matrix ── */
  .matrix-wrap {{ overflow-x:auto; }}
  table.confusion {{ border-collapse:collapse; width:100%; font-size:.88rem; }}
  table.confusion th, table.confusion td {{ padding:10px 12px; text-align:center; }}
  .corner {{ background:var(--bg); font-size:.78rem; color:var(--light); }}
  .col-label {{ background:var(--teal-pale); color:var(--teal); font-weight:700; font-size:.8rem; white-space:nowrap; }}
  .row-label {{ background:var(--teal-pale); color:var(--teal); font-weight:700; font-size:.8rem; text-align:left; white-space:nowrap; }}
  table.confusion td {{ border:1px solid var(--border); min-width:64px; font-size:.95rem; }}
  .cell-pct {{ font-size:.7rem; color:var(--mid); margin-top:2px; }}

  /* ── per-class bars ── */
  table.cls-table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  table.cls-table th {{ text-align:left; padding:8px 12px; background:var(--bg); color:var(--light); font-weight:600; font-size:.78rem; border-bottom:1px solid var(--border); }}
  table.cls-table td {{ padding:10px 12px; border-bottom:1px solid var(--border); }}
  .cls-name {{ font-weight:700; color:var(--dark); width:110px; }}
  .bar-cell {{ width:40%; }}
  .bar-bg {{ background:var(--border); border-radius:4px; height:10px; }}
  .bar-fill {{ height:10px; border-radius:4px; transition:width .5s; }}
  .num {{ text-align:center; color:var(--mid); }}

  /* ── misclassifications ── */
  .wrong-section {{ margin-bottom:24px; }}
  .wrong-section h4 {{ font-size:.95rem; font-weight:700; margin-bottom:10px; display:flex; align-items:center; gap:8px; }}
  .cls-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  table.wrong-table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
  table.wrong-table th {{ background:var(--bg); padding:7px 12px; text-align:left; color:var(--light); font-size:.78rem; border-bottom:1px solid var(--border); }}
  table.wrong-table td {{ padding:7px 12px; border-bottom:1px solid var(--border); }}
  .fname {{ color:var(--mid); font-family:monospace; font-size:.76rem; }}
  .pred-tag {{ background:var(--teal-pale); color:var(--teal); padding:2px 8px; border-radius:12px; font-size:.8rem; font-weight:600; }}

  footer {{ text-align:center; padding:32px; color:var(--light); font-size:.82rem; }}
</style>
</head>
<body>

<header>
  <div class="logo-icon">
    <svg width="26" height="26" fill="none" stroke="white" stroke-width="2.5" viewBox="0 0 24 24">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>
  </div>
  <div>
    <h1>BrainScope AI — Test Report</h1>
    <p>Model: brain-djpb5/3 &nbsp;|&nbsp; Dataset: diagnoct &nbsp;|&nbsp; Split: test &nbsp;|&nbsp; Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  </div>
</header>

<main>

  <!-- stat cards -->
  <div class="stat-cards">
    <div class="stat-card">
      <div class="val">{overall}%</div>
      <div class="lbl">Overall Accuracy</div>
    </div>
    <div class="stat-card">
      <div class="val">{sum(data['per_class'][c]['total'] for c in CLASSES)}</div>
      <div class="lbl">Test Images</div>
    </div>
    <div class="stat-card">
      <div class="val">{len(CLASSES)}</div>
      <div class="lbl">Classes</div>
    </div>
    <div class="stat-card">
      <div class="val">{sum(data['per_class'][c]['total'] - data['per_class'][c]['correct'] for c in CLASSES)}</div>
      <div class="lbl">Misclassified</div>
    </div>
  </div>

  <!-- confusion matrix -->
  <div class="section">
    <h2>Confusion Matrix</h2>
    <p style="color:var(--mid);font-size:.85rem;margin-bottom:16px">Rows = Actual &nbsp;|&nbsp; Columns = Predicted &nbsp;|&nbsp; Green diagonal = correct &nbsp;|&nbsp; Red off-diagonal = errors</p>
    <div class="matrix-wrap">
      <table class="confusion">
        <thead>
          <tr>
            <th class="corner">Actual \\ Predicted</th>
            {col_headers}
          </tr>
        </thead>
        <tbody>{matrix_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- per-class metrics -->
  <div class="section">
    <h2>Per-Class Metrics</h2>
    <table class="cls-table">
      <thead>
        <tr>
          <th>Class</th><th>Accuracy</th><th></th>
          <th>Correct/Total</th><th>Precision</th><th>Recall</th><th>F1</th>
        </tr>
      </thead>
      <tbody>{bar_rows}</tbody>
    </table>
  </div>

  <!-- misclassifications -->
  <div class="section">
    <h2>Misclassified Images</h2>
    {wrong_tables}
  </div>

</main>

<footer>BrainScope AI Test Report &nbsp;|&nbsp; Model brain-djpb5/3 &nbsp;|&nbsp; CC BY 4.0</footer>
</body>
</html>"""

OUT.write_text(html, encoding="utf-8")
print(f"Report saved: {OUT}")
print(f"Overall accuracy: {overall}%")
for cls in CLASSES:
    m = per_class[cls]
    acc = m['tp'] / m['total'] * 100
    print(f"  {cls:<12} {acc:.1f}%  (P:{m['precision']:.1f}% R:{m['recall']:.1f}% F1:{m['f1']:.1f})")
