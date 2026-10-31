#!/usr/bin/env python3
"""
Builds the minimalist GitHub Profile README prototype for Siyuan-Zhu/Siyuan-Zhu
resolving Ticket #14 on the nested Wayfinder sub-map (#7).

Refinements applied:
1. Simple top intro (`### Hi, I'm David` + `<sub>Building in finance, data, and AI</sub>`) —
   no unbuilt website link (`siyuan-zhu.github.io` removed) and no left sidebar.
2. Explanation phrase strictly <= 50 characters:
   "Separate threads locking into shared form." (42 chars)
3. Source attribution:
   - If source photo: platform link (`<a href="https://unsplash.com/photos/...">Photo via Unsplash</a>`), not person name.
   - If SVG fallback: <= 50-char explanation (`SVG generated from daily sentiment contours.` — 44 chars).
4. Simple "How it works" explanation under the Connect 4 game board (inspired by DoubleGremlin181).
"""

from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def build_minimal_svgs() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    # 1. Topographic Horizon SVG Fallback (Pure visual + word, zero HUD/telemetry text)
    topo_svg = textwrap.dedent("""\
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 300" width="100%">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#0d1117"/>
          <stop offset="100%" stop-color="#151b23"/>
        </linearGradient>
        <linearGradient id="horizon" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.06"/>
          <stop offset="50%" stop-color="#818cf8" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="#34d399" stop-opacity="0.08"/>
        </linearGradient>
      </defs>
      <rect width="840" height="300" rx="10" fill="url(#bg)" stroke="#30363d" stroke-width="1"/>
      <path d="M 0 220 Q 210 165 420 205 T 840 180 L 840 300 L 0 300 Z" fill="url(#horizon)"/>
      <path d="M 0 242 Q 260 190 510 228 T 840 206" fill="none" stroke="#58a6ff" stroke-opacity="0.32" stroke-width="1.3"/>
      <path d="M 0 262 Q 200 225 460 250 T 840 232" fill="none" stroke="#a371f7" stroke-opacity="0.25" stroke-width="1.1"/>
      <path d="M 0 280 Q 300 252 560 270 T 840 258" fill="none" stroke="#3fb950" stroke-opacity="0.20" stroke-width="1"/>
      <circle cx="420" cy="136" r="28" fill="none" stroke="#8b949e" stroke-opacity="0.22" stroke-width="1"/>
      <circle cx="420" cy="136" r="3" fill="#e6edf3" fill-opacity="0.7"/>
      <text x="420" y="142" text-anchor="middle" fill="#f0f6fc" font-family="Georgia, 'Times New Roman', serif" font-size="34" font-weight="500" letter-spacing="8">CONVERGENCE</text>
    </svg>
    """)
    (ASSETS / "daily_pulse.svg").write_text(topo_svg, encoding="utf-8")

    # 2. Swiss Editorial Plate SVG Fallback (Pure monochrome plate + word, zero HUD/telemetry text)
    plate_svg = textwrap.dedent("""\
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 260" width="100%">
      <rect width="840" height="260" rx="8" fill="#0d1117" stroke="#30363d" stroke-width="1"/>
      <line x1="60" y1="64" x2="780" y2="64" stroke="#21262d" stroke-width="1"/>
      <line x1="60" y1="196" x2="780" y2="196" stroke="#21262d" stroke-width="1"/>
      <text x="420" y="142" text-anchor="middle" fill="#f0f6fc" font-family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif" font-size="36" font-weight="600" letter-spacing="10">CONVERGENCE</text>
    </svg>
    """)
    (ASSETS / "daily_pulse_plate.svg").write_text(plate_svg, encoding="utf-8")

    # 3. Clean Connect 4 Board SVG (7 columns x 6 rows)
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 2, 1, 2, 0, 0],
        [0, 1, 1, 2, 2, 1, 0],
    ]
    cells_svg = []
    for r in range(6):
        for c in range(7):
            cx = 95 + c * 65
            cy = 72 + r * 54
            val = grid[r][c]
            if val == 0:
                cells_svg.append(f'<circle cx="{cx}" cy="{cy}" r="20" fill="#161b22" stroke="#30363d" stroke-width="1.2"/>')
            elif val == 1:
                cells_svg.append(f'<circle cx="{cx}" cy="{cy}" r="20" fill="#f59e0b" stroke="#fbbf24" stroke-width="1.5"/>')
            else:
                cells_svg.append(f'<circle cx="{cx}" cy="{cy}" r="20" fill="#38bdf8" stroke="#7dd3fc" stroke-width="1.5"/>')

    col_labels = []
    for c in range(7):
        cx = 95 + c * 65
        col_labels.append(
            f'<text x="{cx}" y="34" text-anchor="middle" fill="#8b949e" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12" font-weight="600">{c + 1}</text>'
        )

    c4_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 580 390" width="100%">
  <rect width="580" height="390" rx="10" fill="#0d1117" stroke="#30363d" stroke-width="1"/>
  {''.join(col_labels)}
  {''.join(cells_svg)}
  <circle cx="195" cy="366" r="5" fill="#f59e0b"/>
  <text x="206" y="370" fill="#c9d1d9" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="11">Visitors (Gold) — Your turn</text>
  <circle cx="355" cy="366" r="5" fill="#38bdf8"/>
  <text x="366" y="370" fill="#8b949e" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="11">Bot (Blue)</text>
</svg>
"""
    (ASSETS / "game_board.svg").write_text(c4_svg, encoding="utf-8")


README_MD = """\
<div align="center">

### Hi, I'm David

<sub>Building in finance, data, and AI</sub>

<br><br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/daily_pulse.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/daily_pulse.svg">
  <img alt="CONVERGENCE" src="./assets/daily_pulse.svg" width="740">
</picture>

<br>

**`CONVERGENCE`** — _Separate threads locking into shared form._

<sub><a href="https://unsplash.com/photos/1506744038136-46273834b3fb">Photo via Unsplash</a> · Mood: <a href="https://pulse.siyuanzhu.dev/vote?dir=spot_on">Spot on (18)</a> · <a href="https://pulse.siyuanzhu.dev/vote?dir=too_sunny">Too sunny (3)</a> · <a href="https://pulse.siyuanzhu.dev/vote?dir=too_heavy">Too heavy (4)</a></sub>

</div>

---

<div align="center">

#### Play Connect 4

<img src="./assets/game_board.svg" width="500" alt="Connect 4 Board">

<br>

| [↓ 1](https://github.com/Siyuan-Zhu/Siyuan-Zhu/issues/new?title=connect4%3A+drop+1&body=Just+click+%27Submit+new+issue%27+to+drop+a+piece+in+Column+1.) | [↓ 2](https://github.com/Siyuan-Zhu/Siyuan-Zhu/issues/new?title=connect4%3A+drop+2&body=Just+click+%27Submit+new+issue%27+to+drop+a+piece+in+Column+2.) | [↓ 3](https://github.com/Siyuan-Zhu/Siyuan-Zhu/issues/new?title=connect4%3A+drop+3&body=Just+click+%27Submit+new+issue%27+to+drop+a+piece+in+Column+3.) | [↓ 4](https://github.com/Siyuan-Zhu/Siyuan-Zhu/issues/new?title=connect4%3A+drop+4&body=Just+click+%27Submit+new+issue%27+to+drop+a+piece+in+Column+4.) | [↓ 5](https://github.com/Siyuan-Zhu/Siyuan-Zhu/issues/new?title=connect4%3A+drop+5&body=Just+click+%27Submit+new+issue%27+to+drop+a+piece+in+Column+5.) | [↓ 6](https://github.com/Siyuan-Zhu/Siyuan-Zhu/issues/new?title=connect4%3A+drop+6&body=Just+click+%27Submit+new+issue%27+to+drop+a+piece+in+Column+6.) | [↓ 7](https://github.com/Siyuan-Zhu/Siyuan-Zhu/issues/new?title=connect4%3A+drop+7&body=Just+click+%27Submit+new+issue%27+to+drop+a+piece+in+Column+7.) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |

<sub>last move: <code>@alex-chen</code> dropped in col 4 (bot answered col 5)</sub>

</div>

##### How it works
Each column link records a move and redirects back to this profile (or submits a 1-click issue). Every time the workflow runs via GitHub Actions, it drops the chosen piece, computes the bot's reply, and updates the board. The rest is a regular game of Connect 4.
"""


PROJECTS_MD = """\
### Projects

| Project | Summary | Verified Result |
| :--- | :--- | :--- |
| **[multi-agent-fpa-workspace](https://github.com/Siyuan-Zhu/multi-agent-fpa-workspace)** | Three role-scoped Claude Agent SDK agents (`variance_analyst`, `budget_architect`, `governance_auditor`) build and audit a rolling operating budget from 24 months of synthetic cost-center actuals under written fiscal decision records (ADRs). | Python guardrails enforce write scope, reject uncited numbers, preserve 7/7 locked analyst overrides across re-runs, and replay two recorded runs (clean + tampered row `CC-208` caught and sent back) in CI. |
| **[finance-analytics-warehouse](https://github.com/Siyuan-Zhu/finance-analytics-warehouse)** | Runs 34 monthly closes for a fictional five-store retailer (100% synthetic NIU data) on a DuckDB + dbt star schema, deduplicating multi-export payables (13.32M to 7.33M NIU) and isolating year-end stock-count shrinkage distortions. | Fails closed on unmapped GL accounts and passes 116 automated SQL and Excel checks per close across DuckDB marts and openpyxl workbooks. |
| **[recurring-lifecycle-and-churn-pipeline](https://github.com/Siyuan-Zhu/recurring-lifecycle-and-churn-pipeline)** | Reconciles 14 months of synthetic recurring payments across 4 rails (320 CRM members in 240 accounts): clusters multi-member accounts via Union-Find, fixes UTC/Eastern date shifts and split-payment joins, and collapses 4-step retry/reversal loops. | Removes 3,562 mirror rows (0.00 NIU double-counted), filters 84 apparent churn alerts down to 21 actionable risks, and enforces a persistent 30-day outreach cooldown with regex-checked NIU redaction. |
| **[oecd-aid-finance-llm-classifier](https://github.com/Siyuan-Zhu/oecd-aid-finance-llm-classifier)** | 2023 AidData development finance classifier using FLAN-UL2 few-shot sector prompts, SBERT semantic example selection, Vote-K active annotation, and AWS SageMaker endpoint deployment across OECD CRS purpose taxonomy codes. | Reaches 96.84% best sector accuracy with SBERT-retrieved few-shot exemplars and reproducible offline evaluation harnesses. |
| **[term-deposit-conversion-roi](https://github.com/Siyuan-Zhu/term-deposit-conversion-roi)** | Blocks post-call duration leakage on the public UCI Bank Marketing dataset (41,188 calls), calibrates pre-call logistic regression probabilities, and prices outbound call thresholds against explicit call cost and conversion margin. | Earns ~1.7x the net value of calling everyone on stratified splits, and includes a time-ordered evaluation showing where macro rate shifts break fixed-threshold policies. |

<sub>Earlier coursework & ML (rebuilt with test suites & CI): <a href="https://github.com/Siyuan-Zhu/computer-vision-document-motion-convnet">computer-vision-document-motion-convnet</a> · <a href="https://github.com/Siyuan-Zhu/lowess-boosting-from-scratch">lowess-boosting-from-scratch</a> · <a href="https://github.com/Siyuan-Zhu/heart-disease-risk-model">heart-disease-risk-model</a></sub>
"""


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Siyuan-Zhu / README.md — Minimalist Profile Prototype</title>
<style>
  :root[data-theme="dark"] {
    --bg-canvas: #0d1117;
    --bg-surface: #161b22;
    --bg-subtle: #1f242c;
    --border-default: #30363d;
    --border-muted: #21262d;
    --fg-default: #e6edf3;
    --fg-muted: #8b949e;
    --fg-subtle: #6e7681;
    --accent-fg: #58a6ff;
    --btn-bg: #21262d;
    --btn-hover: #30363d;
    --bar-bg: rgba(22, 27, 34, 0.92);
  }
  :root[data-theme="light"] {
    --bg-canvas: #ffffff;
    --bg-surface: #f6f8fa;
    --bg-subtle: #eaeef2;
    --border-default: #d0d7de;
    --border-muted: #d8dee4;
    --fg-default: #1f2328;
    --fg-muted: #656d76;
    --fg-subtle: #6e7781;
    --accent-fg: #0969da;
    --btn-bg: #f6f8fa;
    --btn-hover: #eaeef2;
    --bar-bg: rgba(246, 248, 250, 0.94);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 0 0 92px 0;
    background: var(--bg-canvas);
    color: var(--fg-default);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }
  a { color: var(--accent-fg); text-decoration: none; }
  a:hover { text-decoration: underline; }

  .gh-top-nav {
    border-bottom: 1px solid var(--border-default);
    background: var(--bg-surface);
    padding: 10px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 13px;
  }

  .gh-shell {
    max-width: 880px;
    margin: 28px auto;
    padding: 0 20px;
  }

  /* README Frame */
  .gh-readme-box {
    border: 1px solid var(--border-default);
    border-radius: 6px;
    background: var(--bg-canvas);
    clip-path: inset(0 round 6px);
  }
  .gh-readme-header {
    padding: 9px 16px;
    border-bottom: 1px solid var(--border-muted);
    background: var(--bg-surface);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    color: var(--fg-muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .markdown-body {
    padding: 36px 40px;
    font-size: 15px;
    line-height: 1.6;
  }
  .markdown-body h3 {
    margin: 0 0 4px 0;
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.2px;
  }
  .markdown-body h4 {
    margin: 0 0 14px 0;
    font-size: 16px;
    font-weight: 600;
  }
  .markdown-body h5 {
    margin: 22px 0 6px 0;
    font-size: 14px;
    font-weight: 600;
  }
  .markdown-body hr {
    border: 0;
    border-top: 1px solid var(--border-muted);
    margin: 28px 0;
  }
  .markdown-body code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 85%;
    padding: 2px 6px;
    background: var(--bg-surface);
    border-radius: 4px;
  }
  .markdown-body table {
    border-collapse: collapse;
    margin: 10px auto;
  }
  .markdown-body th, .markdown-body td {
    border: 1px solid var(--border-default);
    padding: 6px 14px;
    font-size: 13px;
  }
  .markdown-body th {
    background: var(--bg-surface);
    font-weight: 600;
  }
  .center { text-align: center; }

  /* Curated Photo Card */
  .photo-card {
    max-width: 740px;
    margin: 18px auto 0 auto;
    border-radius: 8px;
    clip-path: inset(0 round 8px);
    border: 1px solid var(--border-default);
    background: #0b0f17;
  }
  .photo-surface {
    width: 100%;
    height: 290px;
    background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1400&q=85');
    background-size: cover;
    background-position: center 44%;
  }

  .vote-pill {
    cursor: pointer;
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid transparent;
    transition: all 0.12s;
  }
  .vote-pill:hover {
    border-color: var(--border-default);
    background: var(--bg-surface);
    text-decoration: none;
  }
  .vote-pill.voted {
    border-color: var(--accent-fg);
    background: rgba(88, 166, 255, 0.12);
    font-weight: 600;
  }

  /* Move Table Links */
  .move-table td {
    padding: 5px 12px !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12.5px !important;
  }
  .move-link {
    cursor: pointer;
    font-weight: 600;
  }

  /* Simulated GitHub Pre-Filled Issue Modal */
  .issue-preview {
    max-width: 500px;
    margin: 14px auto 0 auto;
    padding: 12px 14px;
    border-radius: 6px;
    border: 1px solid var(--border-default);
    background: var(--bg-surface);
    text-align: left;
    font-size: 12.5px;
    display: none;
  }
  .issue-preview.open { display: block; }
  .issue-preview-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    color: var(--fg-muted);
    font-size: 11.5px;
  }
  .btn-submit-issue {
    background: #238636;
    color: #ffffff;
    border: 1px solid rgba(240,246,252,0.1);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .btn-submit-issue:hover { background: #2ea043; }

  /* Raw Markdown Inspector */
  .raw-drawer {
    margin-top: 18px;
    border: 1px solid var(--border-default);
    border-radius: 6px;
    background: var(--bg-surface);
    padding: 10px 14px;
  }
  .raw-drawer pre {
    margin: 10px 0 0 0;
    padding: 12px;
    background: var(--bg-canvas);
    border: 1px solid var(--border-muted);
    border-radius: 4px;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    line-height: 1.45;
  }

  /* Bottom Prototype Bar */
  .proto-bar {
    position: fixed;
    bottom: 14px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 999;
    background: var(--bar-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-default);
    border-radius: 999px;
    padding: 7px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.4);
    font-size: 12px;
  }
  .proto-btn {
    border: 1px solid var(--border-default);
    background: var(--btn-bg);
    color: var(--fg-default);
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 12px;
    cursor: pointer;
  }
  .proto-btn.active {
    background: var(--accent-fg);
    color: #fff;
    border-color: var(--accent-fg);
    font-weight: 600;
  }
</style>
</head>
<body>

<div class="gh-top-nav">
  <div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:var(--fg-muted);">
    Siyuan-Zhu / <strong style="color:var(--fg-default);" id="topFileName">README.md</strong>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="color:var(--fg-muted);font-size:12px;">Visual Medium:</span>
    <select id="pulseSelect" onchange="setPulseMedium(this.value)" style="background:var(--btn-bg);color:var(--fg-default);border:1px solid var(--border-default);border-radius:6px;padding:3px 8px;font-size:12px;">
      <option value="photo">1. Curated Photography (Default)</option>
      <option value="topo">2. Topographic Horizon SVG (Fallback)</option>
      <option value="plate">3. Swiss Editorial Plate SVG (Fallback)</option>
    </select>
    <button class="proto-btn" onclick="toggleTheme()" id="themeBtn">Theme: Dark</button>
  </div>
</div>

<div class="gh-shell">
  <div class="gh-readme-box">
    <div class="gh-readme-header">
      <span id="fileHeaderTitle">README.md</span>
      <span style="color:var(--fg-subtle);font-size:11.5px;">42 chars phrase · Platform source credit / SVG explanation</span>
    </div>
    <div class="markdown-body" id="readmeMount"></div>
  </div>

  <details class="raw-drawer">
    <summary style="cursor:pointer;font-size:12px;font-weight:600;color:var(--fg-muted);">
      Inspect raw sanitized Markdown (<code id="rawFileName">README.md</code>)
    </summary>
    <pre><code id="rawMarkdownMount"></code></pre>
  </details>
</div>

<!-- Floating Bottom Bar -->
<div class="proto-bar">
  <span style="color:var(--fg-muted);font-weight:600;">Repo File:</span>
  <button class="proto-btn active" id="btnPageReadme" onclick="setPageView('readme')">README.md</button>
  <button class="proto-btn" id="btnPageProjects" onclick="setPageView('projects')">PROJECTS.md</button>
  <span style="color:var(--border-default);">|</span>
  <span style="color:var(--fg-muted);font-weight:600;">Active Game:</span>
  <select id="gameSelect" onchange="setGameType(this.value)" style="background:var(--btn-bg);color:var(--fg-default);border:1px solid var(--border-default);border-radius:999px;padding:3px 9px;font-size:12px;">
    <option value="connect4">Connect 4 (Default)</option>
    <option value="othello">Reversi / Equilibrium (Rotatable)</option>
  </select>
</div>

<script>
const RAW_README = __RAW_README_JSON__;
const RAW_PROJECTS = __RAW_PROJECTS_JSON__;

const state = {
  page: 'readme',
  theme: 'dark',
  pulseMedium: 'photo',
  gameType: 'connect4',
  votes: { spot_on: 18, too_sunny: 3, too_heavy: 4 },
  userVote: null,
  pendingCol: null,
  turn: 7,
  lastMoveText: '@alex-chen dropped in col 4 (bot answered col 5)',
  c4: [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 2, 1, 2, 0, 0],
    [0, 1, 1, 2, 2, 1, 0]
  ]
};

function renderPulseVisual() {
  const dark = state.theme === 'dark';
  if (state.pulseMedium === 'photo') {
    return '<div class="photo-card"><div class="photo-surface"></div></div>';
  }
  if (state.pulseMedium === 'topo') {
    const bg1 = dark ? '#0d1117' : '#f8fafc';
    const bg2 = dark ? '#151b23' : '#f1f5f9';
    const border = dark ? '#30363d' : '#d0d7de';
    const fg = dark ? '#f0f6fc' : '#0f172a';
    return (
      '<div style="margin-top:18px;">' +
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 280" width="740" style="max-width:100%;border-radius:8px;border:1px solid ' + border + ';">' +
      '<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" stop-color="' + bg1 + '"/><stop offset="100%" stop-color="' + bg2 + '"/>' +
      '</linearGradient></defs>' +
      '<rect width="840" height="280" fill="url(#bg)"/>' +
      '<path d="M 0 210 Q 210 155 420 195 T 840 170" fill="none" stroke="#58a6ff" stroke-opacity="0.35" stroke-width="1.4"/>' +
      '<path d="M 0 232 Q 260 180 510 218 T 840 196" fill="none" stroke="#a371f7" stroke-opacity="0.28" stroke-width="1.2"/>' +
      '<path d="M 0 254 Q 200 215 460 240 T 840 222" fill="none" stroke="#3fb950" stroke-opacity="0.22" stroke-width="1"/>' +
      '<circle cx="420" cy="128" r="26" fill="none" stroke="#8b949e" stroke-opacity="0.22" stroke-width="1"/>' +
      '<circle cx="420" cy="128" r="3" fill="' + fg + '" fill-opacity="0.7"/>' +
      '<text x="420" y="134" text-anchor="middle" fill="' + fg + '" font-family="Georgia, \'Times New Roman\', serif" font-size="32" font-weight="500" letter-spacing="8">CONVERGENCE</text>' +
      '</svg></div>'
    );
  }
  const bg = dark ? '#0d1117' : '#ffffff';
  const border = dark ? '#30363d' : '#d0d7de';
  const line = dark ? '#21262d' : '#eaeef2';
  const fg = dark ? '#f0f6fc' : '#0f172a';
  return (
    '<div style="margin-top:18px;">' +
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 240" width="740" style="max-width:100%;border-radius:8px;border:1px solid ' + border + ';">' +
    '<rect width="840" height="240" fill="' + bg + '"/>' +
    '<line x1="60" y1="58" x2="780" y2="58" stroke="' + line + '" stroke-width="1"/>' +
    '<line x1="60" y1="182" x2="780" y2="182" stroke="' + line + '" stroke-width="1"/>' +
    '<text x="420" y="130" text-anchor="middle" fill="' + fg + '" font-family="-apple-system, BlinkMacSystemFont, \'Inter\', sans-serif" font-size="34" font-weight="600" letter-spacing="10">CONVERGENCE</text>' +
    '</svg></div>'
  );
}

function renderConnect4Svg() {
  const dark = state.theme === 'dark';
  const bg = dark ? '#0d1117' : '#f8fafc';
  const border = dark ? '#30363d' : '#d0d7de';
  const emptyFill = dark ? '#161b22' : '#ffffff';
  const emptyStroke = dark ? '#30363d' : '#cbd5e1';
  const textMain = dark ? '#c9d1d9' : '#334155';
  const textMuted = dark ? '#8b949e' : '#64748b';

  let headers = '';
  for (let c = 0; c < 7; c++) {
    const cx = 95 + c * 65;
    headers += '<text x="' + cx + '" y="34" text-anchor="middle" fill="' + textMuted + '" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12" font-weight="600">' + (c + 1) + '</text>';
  }

  let discs = '';
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 7; c++) {
      const cx = 95 + c * 65;
      const cy = 72 + r * 52;
      const val = state.c4[r][c];
      if (val === 0) {
        discs += '<circle cx="' + cx + '" cy="' + cy + '" r="19" fill="' + emptyFill + '" stroke="' + emptyStroke + '" stroke-width="1.2" style="cursor:pointer" onclick="openMoveIssue(' + (c + 1) + ')"/>';
      } else if (val === 1) {
        discs += '<circle cx="' + cx + '" cy="' + cy + '" r="19" fill="#f59e0b" stroke="#fbbf24" stroke-width="1.5"/>';
      } else {
        discs += '<circle cx="' + cx + '" cy="' + cy + '" r="19" fill="#38bdf8" stroke="#7dd3fc" stroke-width="1.5"/>';
      }
    }
  }

  return (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 580 380" width="500" style="max-width:100%;border-radius:10px;border:1px solid ' + border + ';background:' + bg + ';">' +
    headers + discs +
    '<circle cx="195" cy="358" r="5" fill="#f59e0b"/>' +
    '<text x="206" y="362" fill="' + textMain + '" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="11">Visitors (Gold) — Your turn</text>' +
    '<circle cx="355" cy="358" r="5" fill="#38bdf8"/>' +
    '<text x="366" y="362" fill="' + textMuted + '" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="11">Bot (Blue)</text>' +
    '</svg>'
  );
}

function renderVoteBar() {
  const sCls = state.userVote === 'spot_on' ? 'vote-pill voted' : 'vote-pill';
  const uCls = state.userVote === 'too_sunny' ? 'vote-pill voted' : 'vote-pill';
  const hCls = state.userVote === 'too_heavy' ? 'vote-pill voted' : 'vote-pill';

  let sourceNote = '';
  if (state.pulseMedium === 'photo') {
    sourceNote = '<a href="https://unsplash.com/photos/1506744038136-46273834b3fb" target="_blank" rel="noopener">Photo via Unsplash</a> · ';
  } else if (state.pulseMedium === 'topo') {
    sourceNote = 'SVG generated from daily sentiment contours · ';
  } else {
    sourceNote = 'Minimal SVG plate rendered from daily pulse · ';
  }

  return (
    '<sub>' + sourceNote + 'Mood: ' +
    '<a class="' + sCls + '" onclick="castVote(\'spot_on\'); return false;" href="#vote">Spot on (' + state.votes.spot_on + ')</a> · ' +
    '<a class="' + uCls + '" onclick="castVote(\'too_sunny\'); return false;" href="#vote">Too sunny (' + state.votes.too_sunny + ')</a> · ' +
    '<a class="' + hCls + '" onclick="castVote(\'too_heavy\'); return false;" href="#vote">Too heavy (' + state.votes.too_heavy + ')</a>' +
    '</sub>'
  );
}

function renderReadmePage() {
  const colCells = [1, 2, 3, 4, 5, 6, 7].map(function(col) {
    return '<td><a class="move-link" href="#drop-' + col + '" onclick="openMoveIssue(' + col + '); return false;">↓ ' + col + '</a></td>';
  }).join('');

  const issueModal = state.pendingCol ? (
    '<div class="issue-preview open">' +
      '<div class="issue-preview-header">' +
        '<span>GitHub Move Trigger (<code>Column ' + state.pendingCol + '</code>)</span>' +
        '<a href="#" onclick="state.pendingCol=null; render(); return false;">✕</a>' +
      '</div>' +
      '<div style="margin-bottom:6px;">' +
        '<div style="font-size:11px;color:var(--fg-muted);">Title</div>' +
        '<div style="padding:4px 8px;background:var(--bg-canvas);border:1px solid var(--border-default);border-radius:4px;font-family:monospace;">' +
          'connect4: drop ' + state.pendingCol +
        '</div>' +
      '</div>' +
      '<div style="display:flex;justify-content:flex-end;gap:8px;">' +
        '<button class="btn-submit-issue" onclick="submitMoveIssue(' + state.pendingCol + ')">Play Move (Column ' + state.pendingCol + ')</button>' +
      '</div>' +
    '</div>'
  ) : '';

  const gameHeading = state.gameType === 'connect4'
    ? '<h4>Play Connect 4</h4>'
    : '<h4>Play Reversi (Rotatable Game Slot)</h4>';

  const howItWorks = state.gameType === 'connect4'
    ? 'Each column link records a move and redirects back to this profile (or submits a 1-click issue). Every time the workflow runs via GitHub Actions, it drops the chosen piece, computes the bot\'s reply, and updates the board. The rest is a regular game of Connect 4.'
    : 'Each tile link records a move and redirects back to this profile (or submits a 1-click issue). Every time the workflow runs via GitHub Actions, it flips the captured discs, computes the bot\'s reply, and updates the board.';

  return (
    '<div class="center">' +
      '<h3>Hi, I\'m David</h3>' +
      '<div><sub>Building in finance, data, and AI</sub></div>' +
      renderPulseVisual() +
      '<p style="margin:14px 0 4px 0;">' +
        '<b><code>CONVERGENCE</code></b> — <i>Separate threads locking into shared form.</i>' +
      '</p>' +
      renderVoteBar() +
    '</div>' +
    '<hr>' +
    '<div class="center">' +
      gameHeading +
      renderConnect4Svg() +
      '<table class="move-table"><tr>' + colCells + '</tr></table>' +
      issueModal +
      '<p style="margin:6px 0 0 0;">' +
        '<sub>last move: <code>' + state.lastMoveText + '</code> · <a href="#reset" onclick="resetBoard(); return false;">reset demo</a></sub>' +
      '</p>' +
    '</div>' +
    '<h5>How it works</h5>' +
    '<p style="color:var(--fg-muted);font-size:13.5px;margin:0;">' +
      howItWorks +
    '</p>'
  );
}

function renderProjectsPage() {
  return `
    <h3>Projects</h3>
    <table>
      <thead>
        <tr>
          <th align="left">Project</th>
          <th align="left">Summary</th>
          <th align="left">Verified Result</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><b><a href="https://github.com/Siyuan-Zhu/multi-agent-fpa-workspace">multi-agent-fpa-workspace</a></b></td>
          <td>Three role-scoped Claude Agent SDK agents (<code>variance_analyst</code>, <code>budget_architect</code>, <code>governance_auditor</code>) build and audit a rolling operating budget from 24 months of synthetic cost-center actuals under written fiscal decision records (ADRs).</td>
          <td>Python guardrails enforce write scope, reject uncited numbers, preserve 7/7 locked analyst overrides across re-runs, and replay two recorded runs (clean + tampered row <code>CC-208</code> caught and sent back) in CI.</td>
        </tr>
        <tr>
          <td><b><a href="https://github.com/Siyuan-Zhu/finance-analytics-warehouse">finance-analytics-warehouse</a></b></td>
          <td>Runs 34 monthly closes for a fictional five-store retailer (100% synthetic NIU data) on a DuckDB + dbt star schema, deduplicating multi-export payables (13.32M to 7.33M NIU) and isolating year-end stock-count shrinkage distortions.</td>
          <td>Fails closed on unmapped GL accounts and passes 116 automated SQL and Excel checks per close across DuckDB marts and openpyxl workbooks.</td>
        </tr>
        <tr>
          <td><b><a href="https://github.com/Siyuan-Zhu/recurring-lifecycle-and-churn-pipeline">recurring-lifecycle-and-churn-pipeline</a></b></td>
          <td>Reconciles 14 months of synthetic recurring payments across 4 rails (320 CRM members in 240 accounts): clusters multi-member accounts via Union-Find, fixes UTC/Eastern date shifts and split-payment joins, and collapses 4-step retry/reversal loops.</td>
          <td>Removes 3,562 mirror rows (0.00 NIU double-counted), filters 84 apparent churn alerts down to 21 actionable risks, and enforces a persistent 30-day outreach cooldown with regex-checked NIU redaction.</td>
        </tr>
        <tr>
          <td><b><a href="https://github.com/Siyuan-Zhu/oecd-aid-finance-llm-classifier">oecd-aid-finance-llm-classifier</a></b></td>
          <td>2023 AidData development finance classifier using FLAN-UL2 few-shot sector prompts, SBERT semantic example selection, Vote-K active annotation, and AWS SageMaker endpoint deployment across OECD CRS purpose taxonomy codes.</td>
          <td>Reaches 96.84% best sector accuracy with SBERT-retrieved few-shot exemplars and reproducible offline evaluation harnesses.</td>
        </tr>
        <tr>
          <td><b><a href="https://github.com/Siyuan-Zhu/term-deposit-conversion-roi">term-deposit-conversion-roi</a></b></td>
          <td>Blocks post-call duration leakage on the public UCI Bank Marketing dataset (41,188 calls), calibrates pre-call logistic regression probabilities, and prices outbound call thresholds against explicit call cost and conversion margin.</td>
          <td>Earns ~1.7x the net value of calling everyone on stratified splits, and includes a time-ordered evaluation showing where macro rate shifts break fixed-threshold policies.</td>
        </tr>
      </tbody>
    </table>
    <p><sub>Earlier coursework &amp; ML (rebuilt with test suites &amp; CI): <a href="https://github.com/Siyuan-Zhu/computer-vision-document-motion-convnet">computer-vision-document-motion-convnet</a> · <a href="https://github.com/Siyuan-Zhu/lowess-boosting-from-scratch">lowess-boosting-from-scratch</a> · <a href="https://github.com/Siyuan-Zhu/heart-disease-risk-model">heart-disease-risk-model</a></sub></p>
  `;
}

function castVote(dir) {
  if (state.userVote === dir) return;
  if (state.userVote) state.votes[state.userVote]--;
  state.userVote = dir;
  state.votes[dir]++;
  render();
}

function openMoveIssue(col) {
  state.pendingCol = col;
  render();
}

function submitMoveIssue(col) {
  const c = col - 1;
  let placedRow = -1;
  for (let r = 5; r >= 0; r--) {
    if (state.c4[r][c] === 0) {
      state.c4[r][c] = 1;
      placedRow = r;
      break;
    }
  }
  state.pendingCol = null;
  if (placedRow === -1) {
    render();
    return;
  }
  const order = [3, 2, 4, 1, 5, 0, 6];
  let botCol = -1;
  for (const bc of order) {
    if (bc !== c && state.c4[0][bc] === 0) {
      for (let r = 5; r >= 0; r--) {
        if (state.c4[r][bc] === 0) {
          state.c4[r][bc] = 2;
          botCol = bc + 1;
          break;
        }
      }
      if (botCol !== -1) break;
    }
  }
  state.turn += 1;
  state.lastMoveText = '@you dropped in col ' + col + (botCol !== -1 ? ' (bot answered col ' + botCol + ')' : '');
  render();
}

function resetBoard() {
  state.c4 = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 2, 1, 2, 0, 0],
    [0, 1, 1, 2, 2, 1, 0]
  ];
  state.pendingCol = null;
  state.lastMoveText = '@alex-chen dropped in col 4 (bot answered col 5)';
  render();
}

function setPageView(page) {
  state.page = page;
  document.getElementById('btnPageReadme').classList.toggle('active', page === 'readme');
  document.getElementById('btnPageProjects').classList.toggle('active', page === 'projects');
  document.getElementById('fileHeaderTitle').textContent = page === 'readme' ? 'README.md' : 'PROJECTS.md';
  document.getElementById('topFileName').textContent = page === 'readme' ? 'README.md' : 'PROJECTS.md';
  document.getElementById('rawFileName').textContent = page === 'readme' ? 'README.md' : 'PROJECTS.md';
  render();
}

function setPulseMedium(val) {
  state.pulseMedium = val;
  render();
}

function setGameType(val) {
  state.gameType = val;
  render();
}

function toggleTheme() {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', state.theme);
  document.getElementById('themeBtn').textContent = 'Theme: ' + (state.theme === 'dark' ? 'Dark' : 'Light');
  render();
}

function render() {
  const mount = document.getElementById('readmeMount');
  mount.innerHTML = state.page === 'readme' ? renderReadmePage() : renderProjectsPage();
  document.getElementById('rawMarkdownMount').textContent = state.page === 'readme' ? RAW_README : RAW_PROJECTS;
}

render();
</script>
</body>
</html>
"""


def build_prototype_html(repo_root: Path | None = None) -> Path:
    target_root = repo_root or ROOT
    if target_root == ROOT:
        build_minimal_svgs()
        (target_root / "README.md").write_text(README_MD, encoding="utf-8")
        (target_root / "PROJECTS.md").write_text(PROJECTS_MD, encoding="utf-8")
    html = (
        HTML_CONTENT
        .replace("__RAW_README_JSON__", json.dumps(README_MD))
        .replace("__RAW_PROJECTS_JSON__", json.dumps(PROJECTS_MD))
    )
    out = target_root / "prototype-profile-readme.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    out = build_prototype_html(ROOT)
    print(f"Updated minimalist README.md, PROJECTS.md, and {out}")


if __name__ == "__main__":
    main()
