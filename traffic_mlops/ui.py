"""Flow28 command-deck design system: theme CSS and pure-HTML UI components."""
from __future__ import annotations

import base64
import html
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import streamlit as st


TONE = {
    "good": "#34d399",
    "warn": "#fbbf24",
    "bad": "#fb7185",
    "info": "var(--accent)",
    "violet": "#a78bfa",
}

THEME_DIR = Path(__file__).resolve().parent.parent / "assets" / "themes"

THEMES: dict[str, dict[str, object]] = {
    "command": {
        "label": "Command cyan",
        "hint": "Signal grid, no photo",
        "image": None,
        "accent": "#22d3ee",
        "accent_soft": "#5eead4",
        "accent_2": "#34d399",
        "accent_3": "#a78bfa",
        "accent_ink": "#03141c",
        "ink": "#040a11",
        "panel": "#081420",
        "panel_2": "#0b1a28",
        "text": "#e6f6fb",
        "text_strong": "#f2feff",
        "muted": "#8ba6b5",
        "muted_2": "#637d8b",
    },
    "night-run": {
        "label": "Night run",
        "hint": "Indigo expressway",
        "image": "night-run.jpg",
        "accent": "#8b9bff",
        "accent_soft": "#b7c2ff",
        "accent_2": "#38bdf8",
        "accent_3": "#c084fc",
        "accent_ink": "#070a24",
        "ink": "#050714",
        "panel": "#0a0f22",
        "panel_2": "#111933",
        "text": "#e7e9ff",
        "text_strong": "#f6f5ff",
        "muted": "#9aa3cc",
        "muted_2": "#6f77a3",
    },
    "amber-watch": {
        "label": "Amber watch",
        "hint": "Sodium-lamp dusk",
        "image": "amber-watch.jpg",
        "accent": "#f59e0b",
        "accent_soft": "#fcd34d",
        "accent_2": "#fb923c",
        "accent_3": "#f472b6",
        "accent_ink": "#241404",
        "ink": "#0c0705",
        "panel": "#170f08",
        "panel_2": "#23170d",
        "text": "#fbeede",
        "text_strong": "#fff8ee",
        "muted": "#bda58f",
        "muted_2": "#8e7663",
    },
    "deep-field": {
        "label": "Deep field",
        "hint": "Green corridor",
        "image": "deep-field.jpg",
        "accent": "#a3e635",
        "accent_soft": "#bef264",
        "accent_2": "#34d399",
        "accent_3": "#22d3ee",
        "accent_ink": "#0f1a04",
        "ink": "#04100b",
        "panel": "#08180f",
        "panel_2": "#0c2317",
        "text": "#e6f8ef",
        "text_strong": "#f4fff9",
        "muted": "#93b3a2",
        "muted_2": "#6b8a79",
    },
}

DEFAULT_THEME = "command"


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --ink: #040a11;
  --ink-rgb: 4, 10, 17;
  --panel: #081420;
  --panel-rgb: 8, 20, 32;
  --panel-2: #0b1a28;
  --panel-2-rgb: 11, 26, 40;
  --accent: #22d3ee;
  --accent-rgb: 34, 211, 238;
  --accent-soft: #5eead4;
  --accent-2: #34d399;
  --accent-2-rgb: 52, 211, 153;
  --accent-3: #a78bfa;
  --accent-3-rgb: 167, 139, 250;
  --accent-ink: #03141c;
  --line: rgba(34, 211, 238, .14);
  --text: #e6f6fb;
  --text-strong: #f2feff;
  --muted: #8ba6b5;
  --muted-2: #637d8b;
  --cyan: #22d3ee;
  --mint: #34d399;
  --rose: #fb7185;
  --violet: #a78bfa;
  --font: 'Space Grotesk', 'Bahnschrift', 'Segoe UI Variable Display', 'Segoe UI', system-ui, sans-serif;
  --mono: 'JetBrains Mono', 'Cascadia Code', Consolas, monospace;
}

html, body, [class*="css"] { font-family: var(--font); }

.stApp {
  color: var(--text);
  background-color: var(--ink);
  background-attachment: fixed;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: .6rem; }
.block-container { max-width: 1480px; padding-top: 1.1rem; padding-bottom: 3.2rem; }

h1, h2, h3, h4 { letter-spacing: -.02em; color: var(--text-strong); }
p, li { color: var(--text); }
[data-testid="stCaptionContainer"] p, .stCaption p { color: var(--muted) !important; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(var(--panel-rgb), .99), rgba(var(--ink-rgb), .99));
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: .8rem; }
.side-brand { display: flex; align-items: center; gap: 11px; padding: 4px 2px 14px; border-bottom: 1px solid var(--line); margin-bottom: 14px; }
.side-brand .mark { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; background: linear-gradient(135deg, rgba(var(--accent-rgb), .9), rgba(var(--accent-2-rgb), .9)); color: var(--accent-ink); font-size: 1.05rem; font-weight: 700; box-shadow: 0 0 22px -6px rgba(var(--accent-rgb), .9); }
.side-brand b { display: block; font-size: .95rem; letter-spacing: .07em; color: var(--text-strong); }
.side-brand small { display: block; font-family: var(--mono); font-size: .55rem; letter-spacing: .2em; color: var(--accent-soft); margin-top: 3px; }

[data-testid="stSidebar"] [role="radiogroup"] { gap: .32rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  background: rgba(255, 255, 255, .022);
  border: 1px solid rgba(255, 255, 255, .06);
  border-radius: 11px;
  padding: .5rem .7rem;
  transition: all .18s ease;
  cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { border-color: rgba(var(--accent-rgb), .35); background: rgba(var(--accent-rgb), .06); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(var(--accent-rgb), .18), rgba(var(--accent-2-rgb), .05));
  border-color: rgba(var(--accent-rgb), .55);
  box-shadow: 0 0 0 1px rgba(var(--accent-rgb), .12), 0 10px 26px -18px rgba(var(--accent-rgb), .9);
}
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: .86rem; letter-spacing: .02em; }

[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="input"] > div {
  background: rgba(255, 255, 255, .03) !important;
  border-color: rgba(255, 255, 255, .08) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: rgba(255, 255, 255, .02);
  border: 1px dashed rgba(var(--accent-rgb), .25);
  border-radius: 12px;
  padding: .4rem;
}

/* ---------- controls ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px;
  border: 1px solid rgba(var(--accent-rgb), .38);
  background: linear-gradient(120deg, rgba(var(--accent-rgb), .14), rgba(var(--accent-2-rgb), .08));
  color: var(--text-strong);
  font-weight: 600;
  letter-spacing: .02em;
  transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  border-color: rgba(var(--accent-2-rgb), .8);
  box-shadow: 0 0 26px -8px rgba(var(--accent-rgb), .8);
  transform: translateY(-1px);
  color: #ffffff;
}
[data-testid="stBaseButton-primary"] {
  background: linear-gradient(120deg, var(--accent), var(--accent-2)) !important;
  color: var(--accent-ink) !important;
  border: 0 !important;
}
[data-testid="stBaseButton-primary"]:hover { box-shadow: 0 0 30px -6px rgba(var(--accent-2-rgb), .85) !important; }

[data-testid="stMetric"] {
  border: 1px solid rgba(255, 255, 255, .07);
  border-radius: 14px;
  padding: .75rem .9rem;
  background: linear-gradient(160deg, rgba(var(--panel-2-rgb), .94), rgba(6, 14, 22, .9));
}
[data-testid="stMetricLabel"] p { font-family: var(--mono); font-size: .62rem !important; letter-spacing: .16em; text-transform: uppercase; color: var(--muted-2) !important; }
[data-testid="stMetricValue"] { font-family: var(--mono); color: var(--text-strong); }

[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow: hidden;
  background: rgba(var(--panel-rgb), .6);
}

[data-testid="stAlert"] {
  background: rgba(var(--panel-2-rgb), .8);
  border: 1px solid var(--line);
  border-radius: 12px;
  color: var(--text);
}
[data-testid="stAlert"] p { color: var(--text) !important; }

.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid rgba(255, 255, 255, .06); }
.stTabs [data-baseweb="tab"] {
  background: rgba(255, 255, 255, .025);
  border: 1px solid rgba(255, 255, 255, .06);
  border-radius: 9px 9px 0 0;
  padding: .35rem .9rem;
  color: var(--muted-2);
}
.stTabs [aria-selected="true"] { background: rgba(var(--accent-rgb), .12); border-color: rgba(var(--accent-rgb), .4); color: var(--text-strong); }

[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: rgba(255, 255, 255, .02); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; border-color: rgba(255, 255, 255, .07); background: rgba(255, 255, 255, .018); }

[data-testid="stChatMessage"] {
  background: rgba(255, 255, 255, .028);
  border: 1px solid var(--line);
  border-radius: 14px;
}
[data-testid="stChatInput"] { border-color: rgba(var(--accent-rgb), .25); }

/* ---------- hero ---------- */
.hero {
  position: relative;
  padding: 26px 28px 22px;
  border-radius: 22px;
  border: 1px solid rgba(var(--accent-rgb), .18);
  background:
    linear-gradient(135deg, rgba(var(--accent-rgb), .11), rgba(var(--accent-3-rgb), .07) 46%, rgba(var(--ink-rgb), .3) 78%),
    rgba(var(--ink-rgb), .82);
  overflow: hidden;
  box-shadow: 0 30px 70px -48px rgba(var(--accent-rgb), .9);
}
.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(var(--accent-rgb), .07) 1px, transparent 1px),
    linear-gradient(90deg, rgba(var(--accent-rgb), .07) 1px, transparent 1px);
  background-size: 38px 38px;
  mask-image: radial-gradient(circle at 76% 0%, #000, transparent 72%);
  pointer-events: none;
}
.hero-top { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; position: relative; }
.hero .brand { display: inline-flex; align-items: center; gap: 10px; font-family: var(--mono); font-weight: 600; letter-spacing: .24em; font-size: .8rem; color: var(--text-strong); }
.hero .brand-mark { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: var(--accent-ink); font-size: 1rem; box-shadow: 0 0 22px -6px rgba(var(--accent-rgb), .9); }
.hero .brand-num { color: var(--accent-soft); }
.hero h1 { position: relative; margin: .85rem 0 .3rem; font-size: clamp(1.95rem, 3.8vw, 3rem); font-weight: 700; line-height: 1.02; color: var(--text-strong); }
.hero h1 .grad { background: linear-gradient(100deg, var(--accent) 0%, var(--accent-2) 48%, var(--accent-3) 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-sub { position: relative; margin: 0 0 18px; max-width: 70ch; color: var(--muted); font-size: .95rem; }
.hero-meta { position: relative; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; padding: 12px 0 4px; }
.hero-facts { display: flex; gap: 18px; flex-wrap: wrap; font-family: var(--mono); font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted-2); }
.hero-facts b { color: var(--text-strong); font-weight: 500; }

.chip { display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px; border-radius: 999px; border: 1px solid rgba(255, 255, 255, .1); background: rgba(255, 255, 255, .03); font-family: var(--mono); font-size: .62rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted-2); }
.hero-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip-live { color: #a7f3d0; border-color: rgba(52, 211, 153, .4); background: rgba(52, 211, 153, .08); }
.pulse-dot { width: 6px; height: 6px; border-radius: 50%; background: #34d399; animation: pulse 1.9s infinite; }
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, .6); }
  70% { box-shadow: 0 0 0 7px rgba(52, 211, 153, 0); }
  100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

.signal-cluster { display: inline-flex; gap: 8px; align-items: center; padding: 7px 11px; border-radius: 999px; border: 1px solid rgba(255, 255, 255, .08); background: rgba(var(--ink-rgb), .7); }
.lamp { width: 11px; height: 11px; border-radius: 50%; background: var(--panel-2); box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .08); animation: cycle 6s infinite; }
.lamp.red { --c: #fb7185; animation-delay: 0s; }
.lamp.amber { --c: #fbbf24; animation-delay: 2s; }
.lamp.green { --c: #34d399; animation-delay: 4s; }
@keyframes cycle {
  0%, 28% { background: var(--c); box-shadow: 0 0 12px var(--c); }
  30%, 100% { background: var(--panel-2); box-shadow: none; }
}

.road { position: relative; height: 52px; margin-top: 16px; border-radius: 12px; border: 1px solid rgba(var(--accent-rgb), .12); background: linear-gradient(180deg, rgba(var(--ink-rgb), .92), rgba(var(--panel-rgb), .96)); overflow: hidden; }
.road::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 2px; background: repeating-linear-gradient(90deg, rgba(230, 246, 251, .35) 0 16px, transparent 16px 34px); opacity: .3; }
.car { position: absolute; width: 15px; height: 7px; border-radius: 3px; background: linear-gradient(90deg, var(--accent), rgba(var(--accent-rgb), .4)); box-shadow: 0 0 10px rgba(var(--accent-rgb), .7); animation: drive var(--s, 9s) linear infinite; animation-delay: var(--d, 0s); }
.car:nth-child(odd) { top: 11px; }
.car:nth-child(even) { bottom: 11px; }
.car.b { background: linear-gradient(90deg, var(--accent-2), rgba(var(--accent-2-rgb), .4)); box-shadow: 0 0 10px var(--accent-2); animation-name: drive-rev; }
@keyframes drive { from { left: -6%; } to { left: 104%; } }
@keyframes drive-rev { from { left: 104%; } to { left: -6%; } }

/* ---------- page head / sections ---------- */
.page-head { padding: 6px 0 16px; border-bottom: 1px solid rgba(var(--accent-rgb), .12); margin-bottom: 18px; }
.page-kicker { font-family: var(--mono); font-size: .64rem; letter-spacing: .28em; text-transform: uppercase; color: var(--accent-soft); }
.page-head h1 { margin: .3rem 0 .3rem; font-size: clamp(1.55rem, 2.6vw, 2.05rem); }
.page-sub { margin: 0; color: var(--muted); font-size: .9rem; max-width: 80ch; }

.sec { display: flex; align-items: baseline; gap: 12px; margin: 28px 0 10px; }
.sec .sec-kicker { font-family: var(--mono); font-size: .6rem; letter-spacing: .22em; color: var(--accent-soft); white-space: nowrap; }
.sec h3 { margin: 0; font-size: 1rem; font-weight: 600; color: var(--text-strong); white-space: nowrap; }
.sec .sec-line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(var(--accent-rgb), .28), transparent); }

/* ---------- KPI ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap: 12px; }
.kpi { position: relative; padding: 13px 15px; border-radius: 14px; border: 1px solid rgba(255, 255, 255, .07); background: linear-gradient(160deg, rgba(var(--panel-2-rgb), .94), rgba(6, 14, 22, .86)); overflow: hidden; }
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: var(--tone, var(--accent)); box-shadow: 0 0 14px var(--tone, var(--accent)); }
.k-label { font-family: var(--mono); font-size: .62rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted-2); }
.k-value { font-family: var(--mono); font-size: 1.3rem; font-weight: 600; color: var(--text-strong); margin: .32rem 0 .12rem; letter-spacing: -.01em; }
.k-sub { font-size: .74rem; color: var(--muted); }

/* ---------- ticker ---------- */
.ticker { display: flex; align-items: stretch; margin: 14px 0 4px; border: 1px solid rgba(var(--accent-rgb), .16); border-radius: 12px; background: rgba(var(--panel-rgb), .75); overflow: hidden; }
.ticker-label { flex: none; display: flex; align-items: center; gap: 7px; padding: 8px 12px; border-right: 1px solid rgba(var(--accent-rgb), .18); background: linear-gradient(120deg, rgba(var(--accent-rgb), .18), rgba(var(--accent-2-rgb), .12)); font-family: var(--mono); font-size: .6rem; letter-spacing: .18em; color: var(--text-strong); white-space: nowrap; }
.ticker-view { position: relative; flex: 1; overflow: hidden; display: flex; align-items: center; mask-image: linear-gradient(90deg, transparent, #000 26px, #000 calc(100% - 26px), transparent); }
.ticker-track { display: flex; gap: 36px; white-space: nowrap; width: max-content; animation: scroll 46s linear infinite; padding: 8px 0 8px 36px; }
.ticker:hover .ticker-track { animation-play-state: paused; }
.ticker-item { display: inline-flex; align-items: center; gap: 8px; font-family: var(--mono); font-size: .7rem; letter-spacing: .04em; color: var(--muted-2); }
.ticker-item b { color: var(--text-strong); font-weight: 500; }
.t-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--tone); box-shadow: 0 0 8px var(--tone); }
@keyframes scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }

/* ---------- junction cards ---------- */
.jgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(238px, 1fr)); gap: 12px; }
.jcard { padding: 14px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, .07); background: linear-gradient(170deg, rgba(var(--panel-2-rgb), .94), rgba(var(--ink-rgb), .92)); transition: all .2s ease; }
.jcard:hover { transform: translateY(-2px); border-color: rgba(var(--accent-rgb), .35); box-shadow: 0 20px 44px -30px rgba(var(--accent-rgb), .9); }
.jcard-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.jname { font-weight: 600; font-size: .92rem; letter-spacing: .02em; color: var(--text-strong); }
.pill { font-family: var(--mono); font-size: .56rem; letter-spacing: .12em; text-transform: uppercase; padding: 3px 8px; border-radius: 999px; border: 1px solid var(--tone); color: var(--tone); background: rgba(255, 255, 255, .03); white-space: nowrap; }
.jcard-body { display: flex; gap: 13px; align-items: center; margin: 12px 0 8px; }
.dial { position: relative; flex: none; width: 88px; height: 88px; border-radius: 50%; background: conic-gradient(var(--tone) calc(var(--pct) * 1%), rgba(255, 255, 255, .07) 0); }
.dial::after { content: ""; position: absolute; inset: 7px; border-radius: 50%; background: var(--panel); }
.dial-core { position: absolute; inset: 0; z-index: 1; display: grid; place-content: center; text-align: center; }
.dial-core b { font-family: var(--mono); font-size: .95rem; color: var(--text-strong); line-height: 1; }
.dial-core small { display: block; font-family: var(--mono); font-size: .5rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted-2); margin-top: 3px; }
.jstats { display: grid; gap: 6px; flex: 1; min-width: 0; }
.jstats small { display: block; font-family: var(--mono); font-size: .55rem; letter-spacing: .15em; text-transform: uppercase; color: var(--muted-2); }
.jstats b { font-family: var(--mono); font-size: .78rem; font-weight: 500; color: var(--text-strong); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.jcard-foot { display: flex; justify-content: space-between; align-items: center; gap: 8px; border-top: 1px dashed rgba(255, 255, 255, .08); padding-top: 9px; font-family: var(--mono); font-size: .58rem; letter-spacing: .06em; color: var(--muted-2); }
.spark { width: 108px; height: 30px; flex: none; }

/* ---------- schematic map ---------- */
.map-shell { border: 1px solid rgba(var(--accent-rgb), .14); border-radius: 18px; background: radial-gradient(620px 320px at 50% 0%, rgba(var(--accent-rgb), .07), transparent 72%), rgba(var(--ink-rgb), .72); padding: 6px; }
.jmap { width: 100%; height: auto; display: block; }
.jmap .road-edge { stroke: rgba(var(--accent-rgb), .09); stroke-width: 19; fill: none; stroke-linecap: round; }
.jmap .road { stroke: #17303d; stroke-width: 13; fill: none; stroke-linecap: round; }
.jmap .flow { stroke: var(--accent); stroke-width: 2.2; fill: none; stroke-dasharray: 9 27; opacity: .7; animation: dash 3.4s linear infinite; }
@keyframes dash { to { stroke-dashoffset: -72; } }
.jmap .node-halo { fill: var(--tone); opacity: .15; }
.jmap .node-ring { fill: none; stroke: var(--tone); stroke-width: 1.4; opacity: .7; }
.jmap .node-core { fill: var(--tone); }
.jmap .nlabel { fill: var(--text-strong); font-family: var(--mono); font-size: 9.5px; letter-spacing: .14em; text-anchor: middle; }
.jmap .nvalue { fill: var(--muted); font-family: var(--mono); font-size: 8px; text-anchor: middle; }
.map-legend { display: flex; gap: 16px; flex-wrap: wrap; padding: 8px 6px 2px; font-family: var(--mono); font-size: .62rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.map-legend span { display: inline-flex; align-items: center; gap: 6px; }
.map-legend i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; background: var(--tone); box-shadow: 0 0 8px var(--tone); }

/* ---------- signal card ---------- */
.signal-card { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 16px; border-radius: 14px; border: 1px solid rgba(255, 255, 255, .07); background: linear-gradient(150deg, rgba(var(--panel-2-rgb), .92), rgba(var(--ink-rgb), .9)); }
.signal-box { display: inline-flex; flex-direction: column; gap: 6px; padding: 8px; border-radius: 11px; background: var(--ink); border: 1px solid rgba(255, 255, 255, .08); }
.s-lamp { width: 14px; height: 14px; border-radius: 50%; background: var(--panel-2); box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .06); }
.s-lamp.on-red { background: #fb7185; box-shadow: 0 0 12px #fb7185; }
.s-lamp.on-amber { background: #fbbf24; box-shadow: 0 0 12px #fbbf24; }
.s-lamp.on-green { background: #34d399; box-shadow: 0 0 12px #34d399; }
.signal-read b { display: block; font-family: var(--mono); font-size: 1.5rem; color: var(--text-strong); line-height: 1; }
.signal-read small { display: block; font-family: var(--mono); font-size: .56rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted-2); margin-top: 4px; }
.signal-note { font-size: .7rem; color: var(--muted); max-width: 34ch; }

/* ---------- PSI bars ---------- */
.psi-list { display: grid; gap: 10px; padding: 4px 0; }
.psi-row { display: grid; grid-template-columns: 128px 1fr 58px; gap: 10px; align-items: center; font-family: var(--mono); font-size: .68rem; color: var(--muted-2); }
.psi-track { height: 7px; border-radius: 99px; background: rgba(255, 255, 255, .06); overflow: hidden; }
.psi-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #34d399, #fbbf24, #fb7185); width: var(--w); }
.psi-row b { text-align: right; color: var(--text-strong); font-weight: 500; }

/* ---------- misc ---------- */
.note { color: var(--muted); font-size: .78rem; }
.badge-soft { display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(255, 255, 255, .1); font-family: var(--mono); font-size: .6rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted-2); }
.footer-note { margin-top: 34px; padding-top: 14px; border-top: 1px solid rgba(255, 255, 255, .06); color: var(--muted-2); font-size: .72rem; font-family: var(--mono); letter-spacing: .04em; }

/* ---------- operator profile ---------- */
.profile { display: flex; align-items: center; gap: 18px; padding: 20px 22px; border-radius: 18px; border: 1px solid rgba(var(--accent-rgb), .18); background: linear-gradient(120deg, rgba(var(--accent-rgb), .1), rgba(var(--accent-3-rgb), .06) 52%, rgba(var(--ink-rgb), .4)); box-shadow: 0 24px 60px -46px rgba(var(--accent-rgb), .9); }
.profile-avatar { flex: none; width: 64px; height: 64px; border-radius: 18px; display: grid; place-items: center; font-family: var(--mono); font-size: 1.3rem; font-weight: 600; color: var(--accent-ink); background: linear-gradient(135deg, var(--accent), var(--accent-2)); box-shadow: 0 0 30px -8px rgba(var(--accent-rgb), .9); }
.profile-body { min-width: 0; }
.profile-name { font-size: 1.12rem; font-weight: 600; color: var(--text-strong); letter-spacing: -.01em; }
.profile-id { font-family: var(--mono); font-size: .62rem; letter-spacing: .2em; text-transform: uppercase; color: var(--accent-soft); margin-top: 3px; }
.profile-meta { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 9px; font-family: var(--mono); font-size: .6rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted-2); }
.profile-meta b { color: var(--text-strong); font-weight: 500; }
.profile-tags { margin-left: auto; display: flex; flex-direction: column; gap: 6px; align-items: flex-end; }
@media (max-width: 820px) { .profile { flex-wrap: wrap; } .profile-tags { align-items: flex-start; margin-left: 0; } }

/* ---------- advice list ---------- */
.advice-list { display: grid; gap: 8px; padding: 2px 0; }
.advice { display: flex; gap: 11px; padding: 11px 13px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, .07); border-left: 2px solid var(--tone); background: rgba(255, 255, 255, .022); }
.advice i { flex: none; width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; background: var(--tone); box-shadow: 0 0 9px var(--tone); }
.advice b { display: block; font-size: .84rem; color: var(--text-strong); font-weight: 600; }
.advice span { display: block; margin-top: 3px; font-size: .76rem; color: var(--muted); line-height: 1.45; }

/* ---------- score bars ---------- */
.score-list { display: grid; gap: 10px; padding: 4px 0; }
.score-row { display: grid; grid-template-columns: 176px 1fr 56px; gap: 12px; align-items: center; font-family: var(--mono); font-size: .7rem; color: var(--muted-2); }
.score-row.top { color: var(--text-strong); }
.score-track { height: 9px; border-radius: 99px; background: rgba(255, 255, 255, .06); overflow: hidden; }
.score-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--tone), rgba(255, 255, 255, .22)); width: var(--w); box-shadow: 0 0 12px -3px var(--tone); }
.score-row b { text-align: right; font-weight: 600; color: var(--text-strong); }
.score-note { margin-top: 2px; font-size: .69rem; color: var(--muted-2); grid-column: 1 / -1; }

/* ---------- device rack ---------- */
.dev-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.dev { padding: 14px 15px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, .07); background: linear-gradient(165deg, rgba(var(--panel-2-rgb), .94), rgba(var(--ink-rgb), .9)); }
.dev-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
.dev-name { font-weight: 600; font-size: .88rem; color: var(--text-strong); }
.dev-kind { margin-top: 2px; font-family: var(--mono); font-size: .54rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted-2); }
.dev-state { display: flex; align-items: center; gap: 8px; margin: 12px 0 8px; }
.state-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--tone); box-shadow: 0 0 10px var(--tone); }
.dev-state b { font-family: var(--mono); font-size: .95rem; letter-spacing: .14em; text-transform: uppercase; color: var(--tone); }
.dev-state small { font-family: var(--mono); font-size: .56rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted-2); }
.dev-detail { font-size: .74rem; color: var(--muted); line-height: 1.4; min-height: 2.1em; }
.dev-rec { margin-top: 10px; padding-top: 8px; border-top: 1px dashed rgba(255, 255, 255, .08); font-family: var(--mono); font-size: .58rem; letter-spacing: .1em; text-transform: uppercase; color: #fbbf24; }
.dev-rec.ok { color: #34d399; }

/* ---------- category chips ---------- */
.cat-chips { display: flex; gap: 8px; flex-wrap: wrap; padding: 2px 0; }
.cat-chip { display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px; border-radius: 999px; border: 1px solid rgba(255, 255, 255, .1); background: rgba(255, 255, 255, .025); font-family: var(--mono); font-size: .6rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted-2); }
.cat-chip i { width: 7px; height: 7px; border-radius: 50%; background: var(--tone); box-shadow: 0 0 8px var(--tone); }
.cat-chip b { color: var(--text-strong); font-weight: 600; }

/* ---------- theme swatches ---------- */
.swatches { display: flex; align-items: center; gap: 6px; margin: 2px 0 8px; }
.swatches i { width: 14px; height: 14px; border-radius: 4px; box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .16); }
.swatches small { margin-left: 4px; font-family: var(--mono); font-size: .55rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted-2); line-height: 1.3; }
</style>
"""


_PRIMARY_CSS = """
.stMarkdown a, .stCaption a { color: var(--accent); }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border-color: var(--accent) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] > div { background: var(--accent) !important; }
[data-testid="stBaseButton-primary"] { background: linear-gradient(120deg, var(--accent), var(--accent-2)) !important; color: var(--accent-ink) !important; }
[data-baseweb="select"] [aria-selected="true"] { background-color: rgba(var(--accent-rgb), .16) !important; }
.stTabs [aria-selected="true"] { border-color: rgba(var(--accent-rgb), .4) !important; }
[data-testid="stProgress"] > div > div > div { background-color: var(--accent) !important; }
"""

_MOTION_OFF = """
.stApp *, .stApp *::before, .stApp *::after { animation: none !important; transition: none !important; }
"""

_REDUCED_MOTION = """
@media (prefers-reduced-motion: reduce) {
  .ticker-track, .car, .lamp, .pulse-dot, .jmap .flow { animation: none !important; }
}
"""


def theme_names() -> list[str]:
    return list(THEMES)


def theme_labels() -> dict[str, str]:
    return {name: str(spec["label"]) for name, spec in THEMES.items()}


def theme_preview(theme: str) -> Path | None:
    spec = THEMES.get(theme, THEMES[DEFAULT_THEME])
    image = spec.get("image")
    if not image:
        return None
    path = THEME_DIR / str(image)
    return path if path.is_file() else None


def active_theme() -> str:
    try:
        name = str(st.session_state.get("deck_theme", DEFAULT_THEME))
    except Exception:  # outside a Streamlit script run
        return DEFAULT_THEME
    return name if name in THEMES else DEFAULT_THEME


def _rgb(hex_color: str) -> str:
    value = hex_color.lstrip("#")
    return f"{int(value[0:2], 16)}, {int(value[2:4], 16)}, {int(value[4:6], 16)}"


@lru_cache(maxsize=4)
def _bg_data_uri(filename: str) -> str:
    payload = base64.b64encode((THEME_DIR / filename).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _palette(theme: str) -> dict[str, str]:
    spec = THEMES.get(theme) or THEMES[DEFAULT_THEME]
    return {key: str(value) for key, value in spec.items() if value is not None}


def _root_css(palette: dict[str, str]) -> str:
    accent = palette["accent"]
    vars_ = {
        "--accent": accent,
        "--accent-rgb": _rgb(accent),
        "--accent-soft": palette["accent_soft"],
        "--accent-2": palette["accent_2"],
        "--accent-2-rgb": _rgb(palette["accent_2"]),
        "--accent-3": palette["accent_3"],
        "--accent-3-rgb": _rgb(palette["accent_3"]),
        "--accent-ink": palette["accent_ink"],
        "--ink": palette["ink"],
        "--ink-rgb": _rgb(palette["ink"]),
        "--panel": palette["panel"],
        "--panel-rgb": _rgb(palette["panel"]),
        "--panel-2": palette["panel_2"],
        "--panel-2-rgb": _rgb(palette["panel_2"]),
        "--text": palette["text"],
        "--text-strong": palette["text_strong"],
        "--muted": palette["muted"],
        "--muted-2": palette["muted_2"],
        "--line": f"rgba({_rgb(accent)}, .14)",
        "--primary-color": accent,
    }
    return ":root{" + ";".join(f"{key}:{value}" for key, value in vars_.items()) + "}"


def _app_css(palette: dict[str, str], depth: int) -> str:
    ink = palette["ink"]
    accent, accent_2, accent_3 = palette["accent"], palette["accent_2"], palette["accent_3"]
    layers = [
        f"linear-gradient(rgba({_rgb(accent)}, .035) 1px, transparent 1px)",
        f"linear-gradient(90deg, rgba({_rgb(accent)}, .035) 1px, transparent 1px)",
        f"radial-gradient(900px 520px at 10% -10%, rgba({_rgb(accent)}, .16), transparent 62%)",
        f"radial-gradient(760px 460px at 92% -4%, rgba({_rgb(accent_3)}, .12), transparent 64%)",
        f"radial-gradient(800px 520px at 50% 112%, rgba({_rgb(accent_2)}, .08), transparent 66%)",
    ]
    sizes = ["46px 46px", "46px 46px", "auto", "auto", "auto"]
    repeats = ["repeat", "repeat", "no-repeat", "no-repeat", "no-repeat"]
    image = palette.get("image")
    if image:
        veil = max(0.55, min(0.95, depth / 100))
        layers.append(f"linear-gradient(rgba({_rgb(ink)}, {veil:.2f}), rgba({_rgb(ink)}, {min(0.97, veil + .05):.2f}))")
        layers.append(f'url("{_bg_data_uri(image)}")')
        sizes += ["auto", "cover"]
        repeats += ["no-repeat", "no-repeat"]
    return (
        f".stApp{{background-color:{ink};background-image:{','.join(layers)};"
        f"background-size:{','.join(sizes)};background-repeat:{','.join(repeats)};"
        "background-position:center;background-attachment:fixed;}"
        + (
            f'[data-testid="stSidebar"]{{background:linear-gradient(180deg,'
            f'rgba({_rgb(palette["panel"])}, .96), rgba({_rgb(ink)}, .97));}}'
            if image
            else ""
        )
    )


def inject(theme: str = DEFAULT_THEME, depth: int = 78, motion: bool = True) -> None:
    palette = _palette(theme)
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    blocks = [_root_css(palette), _app_css(palette, depth), _PRIMARY_CSS, _REDUCED_MOTION]
    if not motion:
        blocks.append(_MOTION_OFF)
    st.markdown("<style>" + "".join(blocks) + "</style>", unsafe_allow_html=True)


def theme_swatches(theme: str) -> None:
    palette = _palette(theme)
    dots = "".join(
        f'<i style="background:{palette[key]}"></i>' for key in ("accent", "accent_2", "accent_3")
    )
    st.markdown(
        f'<div class="swatches">{dots}<small>{_esc(palette["hint"])}</small></div>',
        unsafe_allow_html=True,
    )


def _esc(value: object) -> str:
    return html.escape(str(value))


def _tone(value: str) -> str:
    return TONE.get(value, TONE["info"])


def page_head(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<header class="page-head"><div class="page-kicker">{_esc(kicker)}</div>'
        f"<h1>{_esc(title)}</h1><p class=\"page-sub\">{_esc(subtitle)}</p></header>",
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str) -> None:
    st.markdown(
        f'<div class="sec"><span class="sec-kicker">{_esc(kicker)}</span><h3>{_esc(title)}</h3>'
        '<span class="sec-line"></span></div>',
        unsafe_allow_html=True,
    )


def hero(*, version: str, model_ready: bool, junctions: int, observations: int, health: str, scope: str) -> None:
    champion = f"v{version}" if model_ready else "pending"
    health_tone = "good" if health == "Good" else ("warn" if health == "Warning" else "bad")
    st.markdown(
        f"""
<section class="hero">
  <div class="hero-top">
    <div class="brand"><span class="brand-mark">◈</span><span>FLOW<span class="brand-num">28</span></span></div>
    <div class="hero-chips">
      <span class="chip chip-live"><span class="pulse-dot"></span>simulation live</span>
      <span class="chip">mlflow registry</span>
      <span class="chip">project 28</span>
    </div>
  </div>
  <h1>Traffic <span class="grad">Command Deck</span></h1>
  <p class="hero-sub">Four corridors of hourly demand, predicted a day ahead, versioned in the registry, drift-monitored, and served from an offline-capable edge cache.</p>
  <div class="hero-meta">
    <div class="signal-cluster"><span class="lamp red"></span><span class="lamp amber"></span><span class="lamp green"></span></div>
    <div class="hero-facts">
      <span><b>{junctions}</b> corridors</span>
      <span><b>{observations:,}</b> hourly records</span>
      <span>champion <b>{champion}</b></span>
      <span>network <b style="color:{_tone(health_tone)}">{_esc(health)}</b></span>
      <span>scope <b>{_esc(scope)}</b></span>
    </div>
  </div>
  <div class="road" aria-hidden="true">
    <span class="car" style="--s:7.5s;--d:-1s"></span>
    <span class="car b" style="--s:9s;--d:-4s"></span>
    <span class="car" style="--s:11s;--d:-6s"></span>
    <span class="car b" style="--s:8s;--d:-2s"></span>
    <span class="car" style="--s:12.5s;--d:-9s"></span>
    <span class="car b" style="--s:10s;--d:-7s"></span>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def kpi_grid(items: Sequence[dict[str, object]]) -> None:
    cells = "".join(
        f'<div class="kpi" style="--tone:{_tone(str(item.get("tone", "info")))}">'
        f'<div class="k-label">{_esc(item["label"])}</div>'
        f'<div class="k-value">{_esc(item["value"])}</div>'
        f'<div class="k-sub">{_esc(item.get("sub", ""))}</div></div>'
        for item in items
    )
    st.markdown(f'<div class="kpi-grid">{cells}</div>', unsafe_allow_html=True)


def ticker(items: Iterable[tuple[str, str]], label: str = "live feed") -> None:
    rows = list(items)
    if not rows:
        rows = [("good", "No simulated alerts on the current ticks")]
    entries = "".join(
        f'<span class="ticker-item"><i class="t-dot" style="--tone:{_tone(tone)}"></i>'
        f"<b>{_esc(text[:110])}</b></span>"
        for tone, text in rows
    )
    st.markdown(
        f'<div class="ticker"><div class="ticker-label"><span class="pulse-dot"></span>{_esc(label)}</div>'
        f'<div class="ticker-view"><div class="ticker-track">{entries}{entries}</div></div></div>',
        unsafe_allow_html=True,
    )


def sparkline(values: Sequence[float], tone: str = "info", width: int = 108, height: int = 30) -> str:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(numbers) < 2:
        return ""
    low, high = min(numbers), max(numbers)
    span = (high - low) or 1.0
    points = [
        (index / (len(numbers) - 1) * width, height - 3 - (value - low) / span * (height - 6))
        for index, value in enumerate(numbers)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"M0,{height - 1} L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points) + f" L{width},{height - 1} Z"
    color = _tone(tone)
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<path d="{area}" fill="{color}" fill-opacity=".12" stroke="none"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
        "</svg>"
    )


def junction_cards(rows: Sequence[dict[str, object]], history: dict[str, list[float]]) -> None:
    cards = []
    for row in rows:
        occupancy = float(row.get("road_occupancy_pct", 0) or 0)
        online = str(row.get("sensor_status", "")).lower().startswith("online")
        if not online or occupancy >= 78:
            tone, status = "bad", "Critical"
        elif occupancy >= 62:
            tone, status = "warn", "Watch"
        else:
            tone, status = "good", "Nominal"
        trend = history.get(str(row["junction"]), [])
        sensor = _esc(str(row.get("sensor_status", "unknown")))
        dbm = row.get("signal_strength_dbm")
        link = f"5G {float(dbm):.0f} dBm" if isinstance(dbm, (int, float)) else f"5G {_esc(str(row.get('cloud_link', '')))}"
        cards.append(
            f'<article class="jcard"><div class="jcard-head">'
            f'<span class="jname">{_esc(row["junction"])}</span>'
            f'<span class="pill" style="--tone:{_tone(tone)}">{status}</span></div>'
            f'<div class="jcard-body"><div class="dial" style="--pct:{max(0.0, min(occupancy, 100)):.0f};--tone:{_tone(tone)}">'
            f'<div class="dial-core"><b>{occupancy:.0f}</b><small>occupancy</small></div></div>'
            f'<div class="jstats">'
            f'<div><small>flow</small><b>{float(row.get("vehicles_per_hour", 0)):.0f} veh/h</b></div>'
            f'<div><small>speed</small><b>{float(row.get("average_speed_kmh", 0)):.1f} km/h</b></div>'
            f'<div><small>conditions</small><b>{float(row.get("temperature_c", 0)):.1f}°C · {float(row.get("humidity_pct", 0)):.0f}% RH</b></div>'
            f"</div></div>"
            f'<div class="jcard-foot">{sparkline(trend, tone)}'
            f"<span>sensor {sensor} · {link}</span></div>"
            "</article>"
        )
    st.markdown(f'<div class="jgrid">{"".join(cards)}</div>', unsafe_allow_html=True)


def signal_card(occupancy: float, green_seconds: int, mode: str) -> None:
    if occupancy >= 78:
        active, phase = "red", "hold"
    elif occupancy >= 62:
        active, phase = "amber", "balance"
    else:
        active, phase = "green", "release"
    st.markdown(
        f'<div class="signal-card">'
        f'<div class="signal-box">'
        f'<span class="s-lamp{" on-red" if active == "red" else ""}"></span>'
        f'<span class="s-lamp{" on-amber" if active == "amber" else ""}"></span>'
        f'<span class="s-lamp{" on-green" if active == "green" else ""}"></span>'
        f"</div>"
        f'<div class="signal-read"><b>{green_seconds}s</b><small>green phase · {phase}</small></div>'
        f'<div class="signal-note">{_esc(mode)} plan derived from simulated occupancy {occupancy:.0f}%. No signal hardware is contacted.</div>'
        "</div>",
        unsafe_allow_html=True,
    )


CURATED_MAP = {
    "Junction 1": (230.0, 130.0),
    "Junction 2": (660.0, 130.0),
    "Junction 3": (230.0, 300.0),
    "Junction 4": (660.0, 300.0),
}


def _map_positions(names: Sequence[str]) -> tuple[dict[str, tuple[float, float]], bool]:
    if names and all(name in CURATED_MAP for name in names):
        return {name: CURATED_MAP[name] for name in names}, True
    total = max(len(names), 1)
    positions = {}
    for index, name in enumerate(names):
        angle = -math.pi / 2 + index * 2 * math.pi / total
        positions[name] = (450 + 300 * math.cos(angle), 215 + 138 * math.sin(angle))
    return positions, False


def junction_map(rows: Sequence[dict[str, object]]) -> None:
    names = [str(row["junction"]) for row in rows]
    positions, curated = _map_positions(names)

    if curated:
        roads = ["M40 130 H860", "M40 300 H860", "M230 40 V380", "M660 40 V380",
                 "M92 392 C320 244 596 238 812 38"]
    else:
        ring = [positions[name] for name in names]
        roads = []
        for index, (x1, y1) in enumerate(ring):
            x2, y2 = ring[(index + 1) % len(ring)]
            roads.append(f"M{x1:.0f} {y1:.0f} L{x2:.0f} {y2:.0f}")
            roads.append(f"M450 215 L{x1:.0f} {y1:.0f}")

    road_markup = "".join(f'<path class="road-edge" d="{path}"/><path class="road" d="{path}"/>' for path in roads)
    flow_markup = "".join(f'<path class="flow" d="{path}" style="animation-delay:-{index * 0.7:.1f}s"/>' for index, path in enumerate(roads))

    nodes = []
    for row in rows:
        name = str(row["junction"])
        x, y = positions[name]
        occupancy = float(row.get("road_occupancy_pct", 0) or 0)
        online = str(row.get("sensor_status", "")).lower().startswith("online")
        if not online or occupancy >= 78:
            tone, status = "bad", "critical"
        elif occupancy >= 62:
            tone, status = "warn", "watch"
        else:
            tone, status = "good", "nominal"
        nodes.append(
            f'<g style="--tone:{_tone(tone)}">'
            f'<circle class="node-halo" cx="{x:.0f}" cy="{y:.0f}" r="34"/>'
            f'<circle class="node-ring" cx="{x:.0f}" cy="{y:.0f}" r="19"/>'
            f'<circle class="node-core" cx="{x:.0f}" cy="{y:.0f}" r="9"/>'
            f'<text class="nlabel" x="{x:.0f}" y="{y - 34:.0f}">{_esc(name)}</text>'
            f'<text class="nvalue" x="{x:.0f}" y="{y + 40:.0f}">{float(row.get("vehicles_per_hour", 0)):.0f} veh/h · {occupancy:.0f}% · {status}</text>'
            "</g>"
        )

    st.markdown(
        f'<div class="map-shell"><svg class="jmap" viewBox="0 0 900 420" role="img" aria-label="Schematic junction map">'
        f'{road_markup}{flow_markup}{"".join(nodes)}</svg>'
        '<div class="map-legend">'
        f'<span style="--tone:{_tone("good")}"><i></i>nominal</span>'
        f'<span style="--tone:{_tone("warn")}"><i></i>watch</span>'
        f'<span style="--tone:{_tone("bad")}"><i></i>critical</span>'
        "<span>schematic only · no coordinates required</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def psi_bars(features: dict[str, float], threshold: float = 0.2) -> None:
    rows = []
    scale = threshold * 2
    for name, value in features.items():
        width = max(2.0, min(float(value) / scale * 100, 100.0))
        rows.append(
            f'<div class="psi-row"><span>{_esc(name)}</span>'
            f'<div class="psi-track"><div class="psi-fill" style="--w:{width:.1f}%"></div></div>'
            f"<b>{float(value):.3f}</b></div>"
        )
    st.markdown(f'<div class="psi-list">{"".join(rows)}</div>', unsafe_allow_html=True)


def profile_card(profile: dict[str, object]) -> None:
    name = str(profile.get("name") or "Traffic operator").strip() or "Traffic operator"
    parts = [part for part in name.replace("-", " ").split() if part]
    initials = "".join(part[0] for part in parts[:2]).upper() or "OP"
    meta = "".join(
        f"<span>{_esc(label)} <b>{_esc(value)}</b></span>" for label, value in (profile.get("meta") or [])
    )
    tags = "".join(f'<span class="chip">{_esc(tag)}</span>' for tag in (profile.get("tags") or []))
    st.markdown(
        f'<div class="profile"><div class="profile-avatar">{_esc(initials)}</div>'
        f'<div class="profile-body"><div class="profile-name">{_esc(name)}</div>'
        f'<div class="profile-id">{_esc(profile.get("operator_id", "unassigned"))}</div>'
        f'<div class="profile-meta">{meta}</div></div>'
        f'<div class="profile-tags">{tags}</div></div>',
        unsafe_allow_html=True,
    )


def advice_list(items: Sequence[dict[str, object]]) -> None:
    rows = "".join(
        f'<div class="advice" style="--tone:{_tone(str(item.get("tone", "info")))}"><i></i>'
        f'<div><b>{_esc(item.get("label", ""))}</b><span>{_esc(item.get("detail", ""))}</span></div></div>'
        for item in items
    )
    st.markdown(f'<div class="advice-list">{rows}</div>', unsafe_allow_html=True)


def score_bars(items: Sequence[dict[str, object]]) -> None:
    rows = []
    for index, item in enumerate(items):
        score = max(0.0, min(float(item.get("score", 0) or 0), 100.0))
        note = str(item.get("note") or item.get("detail") or "")
        rows.append(
            f'<div class="score-row{" top" if index == 0 else ""}" style="--tone:{_tone(str(item.get("tone", "info")))}">'
            f'<span>{_esc(item.get("name") or item.get("label") or "Plan")}</span>'
            f'<div class="score-track"><div class="score-fill" style="--w:{score:.0f}%"></div></div>'
            f"<b>{score:.0f}%</b>"
            + (f'<div class="score-note">{_esc(note)}</div>' if note else "")
            + "</div>"
        )
    st.markdown(f'<div class="score-list">{"".join(rows)}</div>', unsafe_allow_html=True)


def device_grid(devices: Sequence[dict[str, object]],
                recommendations: dict[str, dict[str, str]] | None = None) -> None:
    recommendations = recommendations or {}
    tiles = []
    for device in devices:
        state = str(device.get("state", "off")).lower()
        mode = str(device.get("mode", "manual"))
        slug = str(device.get("slug", ""))
        is_on = state in {"on", "open", "online"}
        tone = "good" if is_on and mode == "auto" else ("info" if is_on else "warn")
        rec = recommendations.get(slug)
        rec_markup = ""
        if rec:
            target_state = str(rec.get("state", state))
            target_mode = str(rec.get("mode", mode))
            if target_state == state and target_mode == mode:
                rec_markup = '<div class="dev-rec ok">matches recommendation</div>'
            else:
                rec_markup = f'<div class="dev-rec">recommend {_esc(target_state)} · {_esc(target_mode)}</div>'
        tiles.append(
            f'<article class="dev"><div class="dev-top"><div>'
            f'<div class="dev-name">{_esc(device.get("name", slug))}</div>'
            f'<div class="dev-kind">{_esc(device.get("kind", "device"))}</div></div>'
            f'<span class="badge-soft">{_esc(mode)}</span></div>'
            f'<div class="dev-state"><span class="state-dot" style="--tone:{_tone(tone)}"></span>'
            f"<b>{_esc(state)}</b><small>· {_esc(device.get('updated_by', 'system'))}</small></div>"
            f'<div class="dev-detail">{_esc(device.get("detail", ""))}</div>'
            f"{rec_markup}</article>"
        )
    st.markdown(f'<div class="dev-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


def category_chips(items: Sequence[tuple[str, int, str]]) -> None:
    chips = "".join(
        f'<span class="cat-chip" style="--tone:{_tone(tone)}"><i></i>{_esc(name)} <b>{count}</b></span>'
        for name, count, tone in items
    )
    st.markdown(f'<div class="cat-chips">{chips}</div>', unsafe_allow_html=True)


def style_fig(fig, height: int = 320):
    palette = _palette(active_theme())
    grid = f"rgba({_rgb(palette['muted'])}, .18)"
    axis = f"rgba({_rgb(palette['muted'])}, .34)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, Segoe UI, sans-serif", color=palette["muted"], size=12),
        margin=dict(l=6, r=6, t=26, b=6),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.14, x=0, font=dict(size=11)),
        hoverlabel=dict(
            bgcolor=palette["panel_2"],
            bordercolor=f"rgba({_rgb(palette['accent'])}, .4)",
            font=dict(color=palette["text_strong"], family="Space Grotesk, Segoe UI, sans-serif"),
        ),
        colorway=[palette["accent"], palette["accent_2"], "#fbbf24", palette["accent_3"], "#fb7185"],
    )
    fig.update_xaxes(gridcolor=grid, zeroline=False, linecolor=axis)
    fig.update_yaxes(gridcolor=grid, zeroline=False, linecolor=axis)
    return fig
