"""
dashboard/theme.py
------------------
Presentation-only theme for ColdChain Watch Pakistan.
Light "cloud dashboard": aurora canvas, glass panels, navy sidebar,
gradient metric cards. Touches no data, model, RAG agent or pipeline.

THEME BEHAVIOUR
  * Follows Streamlit's own menu (top-right ⋮ > System / Light / Dark).
      Light  -> light dashboard   (default)
      Dark   -> dark dashboard
      System -> follows the device
  * Colours for each mode are defined in .streamlit/config.toml under
    [theme.light] and [theme.dark]. That is what makes Streamlit's own text
    colour flip between dark and light, which is what the detector below reads.
  * Print follows the same mode: light page prints light, dark page prints dark.
  * The mode is read in the browser (a tiny script sets data-cc-theme on <html>),
    so it changes instantly when you click, with no rerun and no Python guessing.

IMPORTANT: _CSS is a Python f-string, so every CSS brace in it is doubled {{ }}.
The other CSS strings are plain strings and use single braces { }.

Usage in app.py (right after st.set_page_config):
    from dashboard.theme import apply_theme, hero, section, footer
    apply_theme()
    hero(city)
"""

import re
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

NAVY = "#0b2e75"
NAVY_DEEP = "#071a47"
INK = "#14295a"
MUTED = "#5d6f9c"

PURPLE = "linear-gradient(135deg, #7c7cff 0%, #5a5ce0 100%)"
TEAL = "linear-gradient(135deg, #1cc8c8 0%, #0a9aa6 100%)"
PINK = "linear-gradient(135deg, #ff7eb0 0%, #e0578c 100%)"
BLUE = "linear-gradient(135deg, #4c94ff 0%, #2563d9 100%)"

# Attribute set on <html> by the script below: "light" or "dark"
_DARK = 'html[data-cc-theme="dark"]'

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea, select {{
    font-family: 'Nunito', 'Segoe UI', sans-serif !important;
}}
html {{ -webkit-text-size-adjust: 100%; }}

/* ================= CANVAS (animated aurora) ================= */
/* NOTE: no "color" set on .stApp on purpose. Streamlit's own text colour there
   is how the script detects Light vs Dark. Text colours are set on .main below. */
.stApp {{
    background:
        radial-gradient(600px 450px at 10% 15%, rgba(255,255,255,.80), transparent 70%),
        radial-gradient(700px 500px at 90% 10%, rgba(130,150,255,.50), transparent 70%),
        radial-gradient(650px 500px at 75% 95%, rgba(255,160,210,.30), transparent 70%),
        radial-gradient(500px 400px at 30% 90%, rgba(120,230,220,.40), transparent 70%),
        linear-gradient(135deg, #8fdcec 0%, #a9e3f5 45%, #d4efff 100%);
    background-attachment: fixed;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{
    padding: 2rem 2.4rem 3rem 2.4rem; max-width: 1450px;
    margin-top: 1rem; margin-bottom: 1rem;
    background: rgba(255,255,255,.50);
    -webkit-backdrop-filter: blur(26px) saturate(1.4);
    backdrop-filter: blur(26px) saturate(1.4);
    border: 1px solid rgba(255,255,255,.80);
    border-radius: 32px;
    box-shadow: 0 24px 70px rgba(20,70,140,.18), inset 0 1px 0 rgba(255,255,255,.95);
    animation: ccfade .6s ease both;
}}
@keyframes ccfade {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

h1, h2, h3, h4 {{ color: {NAVY}; font-weight: 800; letter-spacing: -0.01em; }}
.main p, .main li, .main label {{ color: {INK}; }}
hr {{ border-color: rgba(11,46,117,.10) !important; }}

/* ================= SIDEBAR ================= */
[data-testid="stSidebar"] {{
    background:
        radial-gradient(400px 300px at 0% 0%, rgba(90,120,255,.35), transparent 70%),
        linear-gradient(180deg, {NAVY} 0%, {NAVY_DEEP} 100%) !important;
    border-right: 1px solid rgba(255,255,255,.08);
    box-shadow: 6px 0 30px rgba(7,26,71,.25);
}}

/* kill any dark bands / boxes behind sidebar text */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label * {{
    background: transparent !important;
    background-image: none !important;
}}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #ffffff !important;
    text-shadow: none !important;
}}
[data-testid="stSidebar"] label p {{
    font-weight: 700 !important;
    font-size: .92rem !important;
    letter-spacing: .02em;
    color: #cfe0ff !important;
}}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.18) !important; }}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {{
    color: #e3ecff !important; opacity: 1 !important;
}}

/* sidebar select boxes: solid white, navy text, never faded */
[data-testid="stSidebar"] [data-baseweb="select"] {{
    background: #ffffff !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 16px rgba(0,0,0,.22);
    opacity: 1 !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] div {{
    background: transparent !important;
    border-color: transparent !important;
    border-radius: 14px !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    min-height: 46px;
}}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] input {{
    color: {NAVY} !important;
    -webkit-text-fill-color: {NAVY} !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: {NAVY} !important; opacity: 1 !important; }}

/* dropdown menu (renders outside the sidebar) */
[data-baseweb="popover"] > div {{
    border-radius: 14px !important;
    box-shadow: 0 16px 40px rgba(7,26,71,.30) !important;
    overflow: hidden;
}}
[data-baseweb="popover"] ul,
[data-baseweb="popover"] [role="listbox"] {{ background: #ffffff !important; }}
[data-baseweb="popover"] li,
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li * {{
    color: {NAVY} !important; background: #ffffff !important; font-weight: 600 !important;
}}
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] li:hover *,
[data-baseweb="popover"] [aria-selected="true"],
[data-baseweb="popover"] [aria-selected="true"] * {{
    background: #e6efff !important;
}}

/* slider (History days) */
[data-testid="stSidebar"] [data-testid="stSlider"] {{
    padding-top: .3rem; padding-bottom: 1.2rem;
}}
[data-testid="stSidebar"] [data-testid="stSlider"] * {{ opacity: 1 !important; }}
[data-testid="stSidebar"] [data-testid="stThumbValue"] {{
    background: #ffffff !important;
    color: {NAVY} !important;
    padding: 2px 10px; border-radius: 999px; font-weight: 800 !important;
    box-shadow: 0 4px 10px rgba(0,0,0,.25);
}}
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] {{
    color: #cfe0ff !important; font-weight: 700 !important;
}}
[data-testid="stSidebar"] [role="slider"] {{
    background: #ffffff !important; border: 3px solid #7c8cff !important;
    box-shadow: 0 0 0 6px rgba(124,140,255,.30);
    cursor: grab;
}}
[data-testid="stSidebar"] [role="slider"]:active {{ cursor: grabbing; }}

/* ================= HERO ================= */
.cc-hero {{
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, rgba(255,255,255,.88), rgba(235,244,255,.70));
    -webkit-backdrop-filter: blur(18px); backdrop-filter: blur(18px);
    border-radius: 28px; padding: 28px 34px; margin-bottom: 24px;
    box-shadow: 0 14px 40px rgba(20,60,140,.14), inset 0 1px 0 rgba(255,255,255,.95);
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    border: 1px solid rgba(255,255,255,.9);
}}
.cc-hero::before {{
    content: ""; position: absolute; right: -60px; top: -80px;
    width: 280px; height: 280px; border-radius: 50%;
    background: radial-gradient(circle, rgba(99,120,255,.28), transparent 70%);
}}
.cc-hero::after {{
    content: ""; position: absolute; left: 0; bottom: 0; height: 5px; width: 100%;
    background: linear-gradient(90deg, #6d6ff5, #12b3bd, #f2709f, #3b82f6);
}}
.cc-hero h1 {{ margin: 0; font-size: 2.3rem; color: {NAVY}; line-height: 1.15; position: relative; }}
.cc-hero h1 span {{
    background: linear-gradient(90deg, #4f5df0, #12b3bd);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.cc-hero p {{ margin: 8px 0 0 0; color: {MUTED}; font-size: 1.02rem; font-weight: 600; position: relative; }}
.cc-pill {{
    position: relative; display: inline-flex; align-items: center; gap: 9px;
    background: #ffffff; color: #2563d9; font-weight: 800; font-size: .78rem;
    letter-spacing: .09em; padding: 10px 18px; border-radius: 999px; white-space: nowrap;
    box-shadow: 0 6px 18px rgba(37,99,217,.18);
}}
.cc-dot {{
    width: 10px; height: 10px; border-radius: 50%; background: #16c47f;
    animation: ccpulse 1.8s infinite;
}}
@keyframes ccpulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(22,196,127,.55); }}
    70%  {{ box-shadow: 0 0 0 11px rgba(22,196,127,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(22,196,127,0); }}
}}

/* section headings */
.cc-section {{ margin: 26px 0 12px 0; display: flex; align-items: center; gap: 12px; }}
.cc-section .bar {{
    width: 6px; height: 30px; border-radius: 6px;
    background: linear-gradient(180deg, #6d6ff5, #12b3bd);
}}
.cc-section h3 {{ margin: 0; font-size: 1.45rem; }}
.cc-section small {{ color: {MUTED}; font-weight: 600; display: block; margin-top: 2px; }}

/* window label shown under the History slider selection */
.cc-window {{
    display: inline-block; margin: 0 0 12px 0; padding: 6px 14px; border-radius: 999px;
    background: rgba(79,93,240,.12); color: {NAVY}; font-weight: 800; font-size: .85rem;
}}

/* ================= METRIC CARDS ================= */
[data-testid="stMetric"] {{
    position: relative; overflow: hidden;
    border-radius: 22px; padding: 20px 22px; min-height: 150px;
    color: #ffffff;
    box-shadow: 0 12px 28px rgba(40,80,200,.22), inset 0 1px 0 rgba(255,255,255,.5);
    border: 1px solid rgba(255,255,255,.40);
    transition: transform .2s ease, box-shadow .2s ease;
    background: {BLUE};
}}
[data-testid="stMetric"]::after {{
    content: ""; position: absolute; right: -40px; top: -40px;
    width: 140px; height: 140px; border-radius: 50%;
    background: rgba(255,255,255,.16);
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 20px 40px rgba(40,80,200,.32);
}}
[data-testid="stHorizontalBlock"] > div:nth-child(4n+1) [data-testid="stMetric"] {{ background: {PURPLE}; }}
[data-testid="stHorizontalBlock"] > div:nth-child(4n+2) [data-testid="stMetric"] {{ background: {TEAL}; }}
[data-testid="stHorizontalBlock"] > div:nth-child(4n+3) [data-testid="stMetric"] {{ background: {PINK}; }}
[data-testid="stHorizontalBlock"] > div:nth-child(4n+4) [data-testid="stMetric"] {{ background: {BLUE}; }}
[data-testid="stMetric"] * {{ color: #ffffff !important; }}
[data-testid="stMetricLabel"] p {{
    font-weight: 700; opacity: .95; font-size: .92rem; letter-spacing: .02em;
}}
/* stop values being cut off with "..." */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {{
    font-weight: 900;
    font-size: clamp(1.15rem, 1.7vw, 1.8rem) !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.15 !important;
    word-break: break-word;
}}
[data-testid="stMetricDelta"] {{
    background: rgba(255,255,255,.22); border-radius: 999px;
    padding: 3px 12px; width: fit-content; font-weight: 700; font-size: .8rem;
    white-space: normal !important; overflow: visible !important;
}}
[data-testid="stMetricDelta"] div {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* ================= TABS (segmented premium bar) ================= */
[data-testid="stTabs"] [role="tablist"], div[role="tablist"], [data-baseweb="tab-list"] {{
    display: flex !important; gap: 10px !important; padding: 10px !important;
    width: 100% !important; max-width: 100% !important;
    background: rgba(255,255,255,.85) !important;
    -webkit-backdrop-filter: blur(16px); backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,1) !important;
    border-radius: 26px !important;
    box-shadow: 0 14px 36px rgba(20,60,140,.16), inset 0 1px 0 #fff !important;
    border-bottom: none !important;
    margin-bottom: 22px !important;
}}
[data-testid="stTabs"] [role="tab"], button[role="tab"], button[data-baseweb="tab"], [data-testid="stTab"] {{
    flex: 1 1 0 !important;
    justify-content: center !important;
    height: 56px !important;
    padding: 0 18px !important;
    border-radius: 18px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
    transition: all .22s ease !important;
}}
[role="tab"]::after, [role="tab"]::before {{ border-bottom: none !important; }}
[role="tab"] *, [data-testid="stTab"] * {{
    color: {MUTED} !important; font-weight: 800 !important; font-size: 1.02rem !important;
    border-bottom: none !important;
}}
[role="tab"]:hover {{
    background: linear-gradient(135deg, rgba(91,91,245,.14), rgba(47,109,240,.10)) !important;
    transform: translateY(-2px);
}}
[role="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, #6d6ff5 0%, #2f6df0 55%, #12b3bd 130%) !important;
    box-shadow: 0 12px 26px rgba(60,90,240,.45) !important;
    transform: translateY(-1px);
}}
[role="tab"][aria-selected="true"] * {{ color: #ffffff !important; }}

/* hide default underline / highlight / grey border, any version */
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"],
[role="tablist"] > div:not([role="tab"]), [role="tablist"] > div[aria-hidden="true"] {{
    display: none !important; height: 0 !important; background: transparent !important;
}}

/* icons added by CSS (order: 1 Overview, 2 Alerts, 3 Model, 4 AI, 5 Pipeline) */
[role="tab"]:nth-of-type(1) p::before {{ content: "📊  "; }}
[role="tab"]:nth-of-type(2) p::before {{ content: "🔥  "; }}
[role="tab"]:nth-of-type(3) p::before {{ content: "🎯  "; }}
[role="tab"]:nth-of-type(4) p::before {{ content: "✨  "; }}
[role="tab"]:nth-of-type(5) p::before {{ content: "⚙️  "; }}

/* ================= PANELS ================= */
[data-testid="stPlotlyChart"] {{
    background: rgba(255,255,255,.80); border-radius: 22px; padding: 14px;
    -webkit-backdrop-filter: blur(16px); backdrop-filter: blur(16px);
    box-shadow: 0 10px 28px rgba(20,60,140,.10), inset 0 1px 0 rgba(255,255,255,.95);
    border: 1px solid rgba(255,255,255,.9);
    transition: transform .2s ease, box-shadow .2s ease;
}}
[data-testid="stPlotlyChart"]:hover {{
    transform: translateY(-3px); box-shadow: 0 18px 38px rgba(20,60,140,.16);
}}
[data-testid="stDataFrame"] {{
    background: rgba(255,255,255,.85); border-radius: 18px; padding: 6px;
    box-shadow: 0 10px 28px rgba(20,60,140,.10);
    border: 1px solid rgba(255,255,255,.9); overflow: hidden;
}}
[data-testid="stExpander"] {{
    background: rgba(255,255,255,.72); border-radius: 18px;
    -webkit-backdrop-filter: blur(14px); backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,.9);
    box-shadow: 0 8px 22px rgba(20,60,140,.08);
}}
[data-testid="stExpander"] summary p {{ font-weight: 800; color: {NAVY}; }}
[data-testid="stCode"], pre {{ border-radius: 14px !important; }}

/* alerts */
[data-testid="stAlert"] {{
    border-radius: 18px; border: none;
    box-shadow: 0 8px 22px rgba(20,60,140,.09);
    backdrop-filter: blur(10px);
}}

/* chat (Insight Agent) */
[data-testid="stChatMessage"] {{
    background: rgba(255,255,255,.80); border-radius: 20px; padding: 14px 18px;
    box-shadow: 0 8px 22px rgba(20,60,140,.08); border: 1px solid rgba(255,255,255,.9);
}}
[data-testid="stChatInput"] {{
    border-radius: 999px !important; background: rgba(255,255,255,.9) !important;
    box-shadow: 0 10px 26px rgba(20,60,140,.14);
}}

/* ================= INPUTS & BUTTONS (main area) ================= */
[data-baseweb="input"] > div, [data-baseweb="textarea"] > div,
.main [data-baseweb="select"] > div {{
    background: rgba(255,255,255,.90) !important; border-radius: 14px !important;
    border: 1px solid rgba(180,200,240,.9) !important;
    box-shadow: 0 3px 10px rgba(20,60,140,.06);
}}
.main input {{ color: {INK} !important; }}
.stButton > button, .stDownloadButton > button {{
    border-radius: 14px; font-weight: 800; padding: 10px 28px; border: none;
    background: linear-gradient(135deg, #4f5df0, #2f6df0); color: #ffffff;
    box-shadow: 0 10px 22px rgba(60,90,240,.32); transition: all .18s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-2px); box-shadow: 0 14px 28px rgba(60,90,240,.44); color: #fff;
}}
.stButton > button p, .stDownloadButton > button p {{ color: #ffffff !important; }}

/* ================= CUSTOM KPI CARDS ================= */
.cc-kpis {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 18px; margin-bottom: 22px;
}}
.cc-kpi {{
    position: relative; overflow: hidden; border-radius: 24px; padding: 20px 22px;
    color: #fff; min-height: 150px;
    box-shadow: 0 14px 30px rgba(40,80,200,.24), inset 0 1px 0 rgba(255,255,255,.5);
    border: 1px solid rgba(255,255,255,.4);
    transition: transform .2s ease, box-shadow .2s ease;
    animation: ccfade .6s ease both;
}}
.cc-kpi:hover {{ transform: translateY(-6px); box-shadow: 0 22px 44px rgba(40,80,200,.34); }}
.cc-kpi::after {{
    content: ""; position: absolute; right: -45px; top: -45px; width: 160px; height: 160px;
    border-radius: 50%; background: rgba(255,255,255,.17);
}}
.cc-kpi.purple {{ background: {PURPLE}; }}
.cc-kpi.teal   {{ background: {TEAL}; }}
.cc-kpi.pink   {{ background: {PINK}; }}
.cc-kpi.blue   {{ background: {BLUE}; }}
.cc-kpi .ic {{
    width: 42px; height: 42px; border-radius: 14px; background: rgba(255,255,255,.22);
    display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
    position: absolute; right: 18px; top: 16px; z-index: 1;
}}
.cc-kpi .lb {{ font-weight: 700; font-size: .92rem; opacity: .95; letter-spacing: .02em; }}
.cc-kpi .vl {{
    font-weight: 900; font-size: clamp(1.3rem, 2vw, 2rem); line-height: 1.15;
    margin: 10px 0 12px 0; word-break: break-word;
}}
.cc-kpi .sb {{
    display: inline-block; background: rgba(255,255,255,.22); border-radius: 999px;
    padding: 4px 12px; font-weight: 700; font-size: .8rem;
}}

/* ================= ALERT BANNER ================= */
.cc-alert {{
    display: flex; align-items: center; gap: 22px; border-radius: 26px;
    padding: 22px 28px; margin: 0 0 22px 0; color: #fff;
    box-shadow: 0 16px 36px rgba(20,60,140,.22);
    border: 1px solid rgba(255,255,255,.4);
}}
.cc-alert.danger {{ background: linear-gradient(135deg, #ff5f7e 0%, #d6285a 100%); }}
.cc-alert.warn   {{ background: linear-gradient(135deg, #ffb347 0%, #f2784b 100%); }}
.cc-alert.ok     {{ background: linear-gradient(135deg, #2ed3a0 0%, #0a9aa6 100%); }}
.cc-alert .big   {{ font-size: 2.6rem; }}
.cc-alert h2     {{ margin: 0; color: #fff; font-size: 1.6rem; font-weight: 900; }}
.cc-alert p      {{ margin: 4px 0 0 0; color: rgba(255,255,255,.95); font-weight: 600; }}
.cc-alert .risk  {{ margin-left: auto; min-width: 220px; }}
.cc-alert .risk small {{ font-weight: 800; letter-spacing: .06em; opacity: .95; }}
.cc-alert .track {{
    height: 12px; border-radius: 999px; background: rgba(255,255,255,.30); margin-top: 8px; overflow: hidden;
}}
.cc-alert .fill  {{ height: 100%; border-radius: 999px; background: #ffffff; }}
.cc-alert.danger {{ animation: ccglow 2.2s infinite; }}
@keyframes ccglow {{
    0%, 100% {{ box-shadow: 0 16px 36px rgba(214,40,90,.30); }}
    50%      {{ box-shadow: 0 16px 50px rgba(214,40,90,.55); }}
}}

/* ================= SIDEBAR STATUS CARD ================= */
.cc-status {{
    background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18);
    border-radius: 18px; padding: 14px 16px; margin: 8px 0 14px 0;
}}
.cc-status b {{ color: #ffffff; }}
.cc-status span {{ color: #cfe0ff; font-size: .85rem; font-weight: 600; }}

/* footer */
.cc-footer {{
    text-align: center; color: {MUTED}; font-weight: 600; font-size: .85rem;
    margin-top: 30px; padding-top: 14px; border-top: 1px solid rgba(11,46,117,.10);
}}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: #b9ccf0; border-radius: 10px; }}

/* =====================================================================
   RESPONSIVE: tablet and phone. Keep this at the END of the stylesheet
   so it overrides the desktop rules above.
   ===================================================================== */

/* ---------- Tablet (<= 900px) ---------- */
@media screen and (max-width: 900px) {{
    .stApp {{ background-attachment: scroll; }}   /* fixed bg is janky on iOS */
    .block-container {{
        padding: 1.6rem 1.4rem 2.5rem 1.4rem;
        border-radius: 26px;
    }}
    .cc-hero h1 {{ font-size: 1.9rem; }}
}}

/* ---------- Phone (<= 768px) ---------- */
@media screen and (max-width: 768px) {{

    /* page shell: use the full screen, lighter blur for speed */
    .block-container {{
        padding: 3rem .75rem 2rem .75rem !important;
        margin: .35rem .3rem !important;
        max-width: 100% !important;
        border-radius: 22px !important;
        -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
    }}
    h1 {{ font-size: 1.55rem; }}
    h2 {{ font-size: 1.3rem; }}
    h3 {{ font-size: 1.12rem; }}

    /* hero: stack title above the LIVE pill */
    .cc-hero {{
        flex-direction: column; align-items: flex-start;
        padding: 20px 18px 22px 18px; gap: 14px;
        border-radius: 22px; margin-bottom: 16px;
    }}
    .cc-hero h1 {{ font-size: 1.55rem; }}
    .cc-hero p  {{ font-size: .9rem; margin-top: 6px; }}
    .cc-pill    {{ font-size: .7rem; padding: 8px 14px; }}
    .cc-hero::before {{ width: 180px; height: 180px; right: -50px; top: -60px; }}

    /* section headings */
    .cc-section {{ margin: 18px 0 10px 0; }}
    .cc-section .bar {{ height: 24px; }}
    .cc-section h3 {{ font-size: 1.15rem; }}

    /* tabs: one swipeable row, no squeezing */
    [data-testid="stTabs"] [role="tablist"], div[role="tablist"], [data-baseweb="tab-list"] {{
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        justify-content: flex-start !important;
        gap: 6px !important;
        padding: 6px !important;
        border-radius: 18px !important;
        margin-bottom: 16px !important;
        scrollbar-width: none;
        -webkit-overflow-scrolling: touch;
    }}
    [data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar,
    div[role="tablist"]::-webkit-scrollbar,
    [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}

    [data-testid="stTabs"] [role="tab"], button[role="tab"], button[data-baseweb="tab"], [data-testid="stTab"] {{
        flex: 0 0 auto !important;
        height: 46px !important;
        padding: 0 14px !important;
        border-radius: 14px !important;
        white-space: nowrap !important;
    }}
    [role="tab"] *, [data-testid="stTab"] * {{
        font-size: .9rem !important;
        white-space: nowrap !important;
    }}

    /* Streamlit metric cards */
    [data-testid="stMetric"] {{
        min-height: 0; padding: 14px 16px; border-radius: 18px;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {{ font-size: 1.35rem !important; }}
    [data-testid="stMetricLabel"] p {{ font-size: .82rem; }}

    /* columns stack full width */
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        gap: .75rem !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }}

    /* custom KPI cards: 2 per row, compact */
    .cc-kpis {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px; margin-bottom: 16px;
    }}
    .cc-kpi {{ min-height: 118px; padding: 14px 14px; border-radius: 18px; }}
    .cc-kpi .ic {{ width: 32px; height: 32px; font-size: 1rem; border-radius: 10px; right: 10px; top: 10px; }}
    .cc-kpi .lb {{ font-size: .78rem; padding-right: 34px; }}
    .cc-kpi .vl {{ font-size: 1.2rem; margin: 8px 0 8px 0; }}
    .cc-kpi .sb {{ font-size: .7rem; padding: 3px 9px; }}

    /* alert banner: stack, risk bar goes full width */
    .cc-alert {{
        flex-wrap: wrap; gap: 10px 14px;
        padding: 16px 16px; border-radius: 20px; margin-bottom: 16px;
    }}
    .cc-alert .big  {{ font-size: 2rem; }}
    .cc-alert h2    {{ font-size: 1.15rem; }}
    .cc-alert p     {{ font-size: .88rem; }}
    .cc-alert .risk {{ margin-left: 0; min-width: 0; width: 100%; }}

    /* charts, tables, expanders */
    [data-testid="stPlotlyChart"] {{
        padding: 6px; border-radius: 16px; max-width: 100%;
        -webkit-backdrop-filter: none; backdrop-filter: none;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        overflow-x: auto !important; border-radius: 14px;
    }}
    [data-testid="stExpander"] {{ border-radius: 14px; }}
    [data-testid="stChatMessage"] {{ padding: 10px 12px; border-radius: 16px; }}
    iframe, [data-testid="stImage"] img {{ max-width: 100% !important; }}

    /* thumb-friendly buttons and inputs */
    .stButton > button, .stDownloadButton > button {{
        width: 100%; min-height: 46px; padding: 10px 16px;
    }}
    .main [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{ min-height: 46px; }}

    .cc-footer {{ font-size: .78rem; margin-top: 20px; }}
}}

/* ---------- Very small phones (<= 380px): KPI cards 1 per row ---------- */
@media screen and (max-width: 380px) {{
    .cc-kpis {{ grid-template-columns: 1fr; }}
    .cc-hero h1 {{ font-size: 1.35rem; }}
    [role="tab"] *, [data-testid="stTab"] * {{ font-size: .82rem !important; }}
}}

/* ---------- Touch screens: no hover jumps ---------- */
@media (hover: none) {{
    [data-testid="stMetric"]:hover,
    .cc-kpi:hover,
    [data-testid="stPlotlyChart"]:hover,
    .stButton > button:hover,
    .stDownloadButton > button:hover {{ transform: none; }}
    [role="tab"]:hover {{ transform: none; }}
}}
</style>
"""


# NOTE: everything below is PLAIN strings (not f-strings), so CSS uses SINGLE braces { }.

# ---------------------------------------------------------------------
# DARK (screen). Written as normal rules; _scope() prefixes every selector
# with html[data-cc-theme="dark"] so it only applies when Dark is selected.
# IMPORTANT: never set "color" on .stApp here, the detector reads it.
# ---------------------------------------------------------------------
_DARK_SCREEN_RULES = """
.stApp {
    background:
        radial-gradient(700px 500px at 88% 8%, rgba(90,110,255,.30), transparent 70%),
        radial-gradient(650px 500px at 12% 95%, rgba(18,179,189,.20), transparent 70%),
        radial-gradient(500px 400px at 70% 90%, rgba(255,110,170,.12), transparent 70%),
        linear-gradient(135deg, #050d24 0%, #0a1a44 55%, #0c2552 100%) !important;
}
.block-container {
    background: rgba(12,24,58,.62) !important;
    border-color: rgba(120,150,255,.18) !important;
    box-shadow: 0 24px 70px rgba(0,0,0,.50), inset 0 1px 0 rgba(255,255,255,.06) !important;
}
.block-container h1, .block-container h2,
.block-container h3, .block-container h4 { color: #f1f5ff !important; }
.block-container p, .block-container li, .block-container label,
.block-container [data-testid="stMarkdownContainer"] { color: #d5e0fb !important; }
hr { border-color: rgba(150,175,255,.18) !important; }

/* gradient cards stay white-on-colour */
.block-container [data-testid="stMetric"] *,
.block-container .cc-kpi *,
.block-container .cc-alert * { color: #ffffff !important; }

/* hero */
.block-container .cc-hero {
    background: linear-gradient(135deg, rgba(22,40,90,.92), rgba(13,28,66,.80)) !important;
    border-color: rgba(120,150,255,.22) !important;
    box-shadow: 0 14px 40px rgba(0,0,0,.40), inset 0 1px 0 rgba(255,255,255,.06) !important;
}
.block-container .cc-hero h1 { color: #ffffff !important; }
.block-container .cc-hero p  { color: #a9bbe8 !important; }
.cc-pill { background: #0f2252 !important; color: #8cb8ff !important;
           box-shadow: 0 6px 18px rgba(0,0,0,.35) !important; }
.cc-section small { color: #9fb2e0 !important; }
.cc-window { background: rgba(124,140,255,.20) !important; color: #dbe5ff !important; }
.cc-footer { color: #9fb2e0 !important; border-top-color: rgba(150,175,255,.18) !important; }

/* tabs */
[data-testid="stTabs"] [role="tablist"], div[role="tablist"], [data-baseweb="tab-list"] {
    background: rgba(12,26,62,.88) !important;
    border-color: rgba(120,150,255,.20) !important;
    box-shadow: 0 14px 36px rgba(0,0,0,.40), inset 0 1px 0 rgba(255,255,255,.05) !important;
}
[role="tab"] *, [data-testid="stTab"] * { color: #aebfe8 !important; }
[role="tab"]:hover { background: rgba(110,130,255,.18) !important; }
[role="tab"][aria-selected="true"] * { color: #ffffff !important; }

/* panels */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"],
[data-testid="stExpander"], [data-testid="stChatMessage"] {
    background: rgba(12,26,62,.78) !important;
    border-color: rgba(120,150,255,.18) !important;
    box-shadow: 0 10px 28px rgba(0,0,0,.35) !important;
}
[data-testid="stExpander"] summary p { color: #e6edff !important; }

/* inputs */
[data-baseweb="input"] > div, [data-baseweb="textarea"] > div,
.main [data-baseweb="select"] > div {
    background: rgba(16,32,76,.92) !important;
    border-color: rgba(120,150,255,.30) !important;
}
.main input, .main textarea { color: #e6edff !important; }
[data-testid="stChatInput"] { background: rgba(16,32,76,.92) !important; }

/* dropdown / menu popovers */
[data-baseweb="popover"] ul,
[data-baseweb="popover"] [role="listbox"] { background: #10234f !important; }
[data-baseweb="popover"] li,
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li * { color: #e6edff !important; background: #10234f !important; }
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] li:hover *,
[data-baseweb="popover"] [aria-selected="true"],
[data-baseweb="popover"] [aria-selected="true"] * { background: #1d3a7a !important; }

::-webkit-scrollbar-thumb { background: #2a3f7a; }
"""

# ---------------------------------------------------------------------
# PRINT, shared layout (both modes): A4, hide chrome, print all tabs
# ---------------------------------------------------------------------
_PRINT_LAYOUT = """
<style>
.cc-print-meta { display: none; }

@page { size: A4 portrait; margin: 12mm; }

@media print {
    * {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
        animation: none !important;
        transition: none !important;
    }

    /* Streamlit scrolls inside a fixed-height box; without this only page 1 prints */
    html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main, .main {
        height: auto !important; min-height: 0 !important;
        overflow: visible !important; position: static !important;
    }

    /* hide everything that is not report content */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"], header[data-testid="stHeader"],
    [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stBottom"],
    [data-testid="stChatInput"], [data-baseweb="tab-list"], [role="tablist"],
    .stButton, .stDownloadButton, footer, .cc-pill {
        display: none !important;
    }

    /* flat page */
    .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 100% !important; width: 100% !important;
        margin: 0 !important; padding: 0 !important;
        border: none !important;
        border-radius: 0 !important; box-shadow: none !important;
        -webkit-backdrop-filter: none !important; backdrop-filter: none !important;
    }

    /* print EVERY tab one after another */
    [role="tabpanel"], [data-baseweb="tab-panel"] {
        display: block !important; visibility: visible !important;
        height: auto !important; opacity: 1 !important;
        margin: 0 0 16px 0 !important; padding: 0 !important;
    }

    /* hero becomes the report title block */
    .cc-hero {
        box-shadow: none !important; padding: 16px 18px !important;
        margin-bottom: 14px !important; break-inside: avoid;
    }
    .cc-hero::before { display: none !important; }
    .cc-hero h1 { font-size: 1.6rem !important; }
    .cc-print-meta { display: block !important; margin-top: 6px; font-size: .8rem; font-weight: 700; }

    /* keep cards, charts and tables whole, no shadows or blur */
    .cc-kpi, .cc-alert, [data-testid="stMetric"], [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"], [data-testid="stExpander"], [data-testid="stAlert"],
    [data-testid="stChatMessage"], .cc-section {
        break-inside: avoid; page-break-inside: avoid;
        box-shadow: none !important;
        -webkit-backdrop-filter: none !important; backdrop-filter: none !important;
    }
    .cc-section { break-after: avoid; page-break-after: avoid; margin: 14px 0 8px 0 !important; }

    /* KPI cards in a tidy row of 4 */
    .cc-kpis { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; gap: 8px !important; margin-bottom: 12px !important; }
    .cc-kpi { min-height: 88px !important; padding: 12px 14px !important; }
    .cc-kpi .ic { display: none !important; }
    .cc-kpi .vl { font-size: 1.25rem !important; margin: 6px 0 !important; }
    [data-testid="stMetric"] { min-height: 0 !important; padding: 12px 14px !important; }

    .cc-alert { padding: 12px 16px !important; margin-bottom: 12px !important; }
    .cc-alert.danger { animation: none !important; }
    .cc-footer { margin-top: 18px !important; }
}
</style>
"""

# PRINT, LIGHT colours (default). Plain rules, wrapped in @media print by _build_extra_css().
_PRINT_LIGHT_RULES = """
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main, .main { background: #ffffff !important; }
.block-container, [data-testid="stMainBlockContainer"] { background: #ffffff !important; }
h1, h2, h3, h4 { color: #0b2e75 !important; }
p, li, label { color: #14295a !important; }
.cc-hero { background: #f3f7ff !important; border: 1px solid #d3def5 !important; }
.cc-hero h1 span { color: #2f4de0 !important; -webkit-text-fill-color: #2f4de0 !important; background: none !important; }
.cc-hero p { color: #5d6f9c !important; }
.cc-print-meta { color: #5d6f9c !important; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"],
[data-testid="stExpander"], [data-testid="stChatMessage"] {
    background: #ffffff !important; border: 1px solid #dfe7f7 !important;
}
"""

# PRINT, DARK colours. Scoped to Dark mode only, so a dark page prints dark.
# (Turn on "Background graphics" in the browser print dialog.)
_PRINT_DARK_RULES = """
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main, .main { background: #0b1220 !important; }
.block-container, [data-testid="stMainBlockContainer"] { background: #0b1220 !important; }
h1, h2, h3, h4 { color: #f1f5ff !important; }
p, li, label, [data-testid="stMarkdownContainer"] { color: #d5e0fb !important; }
.cc-hero { background: #13244f !important; border: 1px solid #2a3f7a !important; }
.cc-hero h1 { color: #ffffff !important; }
.cc-hero h1 span { color: #8cb8ff !important; -webkit-text-fill-color: #8cb8ff !important; background: none !important; }
.cc-hero p, .cc-print-meta, .cc-section small { color: #a9bbe8 !important; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"],
[data-testid="stExpander"], [data-testid="stChatMessage"] {
    background: #0f1e45 !important; border: 1px solid #2a3f7a !important;
}
.cc-kpi *, .cc-alert *, [data-testid="stMetric"] * { color: #ffffff !important; }
.cc-footer { color: #9fb2e0 !important; border-top-color: #2a3f7a !important; }
"""


def _scope(rules: str, prefix: str) -> str:
    """Prefix every selector in a block of flat CSS rules with `prefix`."""
    rules = re.sub(r"/\*.*?\*/", "", rules, flags=re.S)

    def one(m):
        sels = []
        for s in m.group(1).split(","):
            s = s.strip()
            if not s:
                continue
            sels.append(prefix if s == "html" else f"{prefix} {s}")
        return ",\n".join(sels) + " {" + m.group(2) + "}"

    return re.sub(r"([^{}]+)\{([^{}]*)\}", one, rules)


def _build_extra_css() -> str:
    dark_screen = "@media screen {\n" + _scope(_DARK_SCREEN_RULES, _DARK) + "\n}"
    print_light = "@media print {\n" + _PRINT_LIGHT_RULES + "\n}"
    print_dark = "@media print {\n" + _scope(_PRINT_DARK_RULES, _DARK) + "\n}"
    return f"<style>\n{dark_screen}\n{print_light}\n{print_dark}\n</style>"


# ---------------------------------------------------------------------
# Browser-side theme detector. Streamlit's own menu (System / Light / Dark)
# sets the text colour on .stApp. Light text = Dark theme. We copy that to
# <html data-cc-theme="light|dark"> so our CSS (screen AND print) follows it.
#
# This works because .streamlit/config.toml defines BOTH [theme.light] and
# [theme.dark]; in Dark, Streamlit's textColor becomes light. (Before, only one
# set of colours existed, so the text stayed dark in Dark mode and the detector
# always answered "light".)
#
# The colour is normalised through a <canvas>, so it works for ANY css colour
# format (rgb, color(srgb ...), oklch ...).
# ---------------------------------------------------------------------
_THEME_JS = r"""
<script>
(function () {
  var P = (window.parent && window.parent !== window) ? window.parent : window;
  var doc = P.document;

  // canvas turns ANY css colour format into plain r,g,b
  var cv = doc.createElement('canvas'); cv.width = cv.height = 1;
  var cx = cv.getContext('2d', { willReadFrequently: true });
  function rgb(css) {
    cx.clearRect(0, 0, 1, 1);
    cx.fillStyle = '#000000';
    cx.fillStyle = css;
    cx.fillRect(0, 0, 1, 1);
    var d = cx.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2]];
  }

  function apply() {
    var el = doc.querySelector('.stApp') || doc.body;
    if (!el) return;
    var c = rgb(P.getComputedStyle(el).color);
    var lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    var mode = lum > 140 ? 'dark' : 'light';      // light text = Dark theme
    var root = doc.documentElement;
    if (root.getAttribute('data-cc-theme') !== mode) {
      root.setAttribute('data-cc-theme', mode);
    }
  }

  if (P.__ccThemeTimer) { P.clearInterval(P.__ccThemeTimer); }
  apply();
  P.__ccThemeTimer = P.setInterval(apply, 250);
  P.addEventListener('beforeprint', apply);        // print follows the mode
  P.addEventListener('focus', apply);
  P.addEventListener('storage', apply);
  doc.addEventListener('visibilitychange', apply);

  // also react to the device switching light/dark while "System" is selected
  try {
    var mq = P.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) { mq.addEventListener('change', apply); }
  } catch (e) {}
})();
</script>
"""


def inject_theme() -> None:
    """Inject the global stylesheet + theme detector. Call once, right after set_page_config."""
    st.markdown(_CSS + _build_extra_css() + _PRINT_LAYOUT, unsafe_allow_html=True)
    components.html(_THEME_JS, height=0)


def hero(city: str = "Karachi") -> None:
    """Top banner replacing the plain title."""
    stamp = datetime.now().strftime("%d %b %Y, %H:%M")
    st.markdown(
        f"""
        <div class="cc-hero">
          <div>
            <h1>ColdChain <span>Watch</span> Pakistan ❄️</h1>
            <p>Live heat exposure &amp; price outlook for cold-store managers · {city}</p>
            <div class="cc-print-meta">Report generated {stamp}</div>
          </div>
          <div class="cc-pill"><span class="cc-dot"></span>LIVE MONITORING</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    """Styled section heading with an accent bar (optional helper)."""
    cap = f"<small>{caption}</small>" if caption else ""
    st.markdown(
        f'<div class="cc-section"><div class="bar"></div>'
        f"<div><h3>{title}</h3>{cap}</div></div>",
        unsafe_allow_html=True,
    )


def window_badge(text: str) -> None:
    """Small pill that shows which date range the History slider is currently showing."""
    st.markdown(f'<div class="cc-window">{text}</div>', unsafe_allow_html=True)


def footer(text: str = "ColdChain Watch Pakistan · Built for the hackathon") -> None:
    """Small footer line (optional helper)."""
    st.markdown(f'<div class="cc-footer">{text}</div>', unsafe_allow_html=True)


def kpi_row(cards: list) -> None:
    """Row of gradient KPI cards that never truncate.
    cards = [dict(label=, value=, sub=, icon=, tone='purple|teal|pink|blue'), ...]
    """
    html = ""
    for c in cards:
        html += (
            f'<div class="cc-kpi {c.get("tone", "blue")}">'
            f'<div class="ic">{c.get("icon", "")}</div>'
            f'<div class="lb">{c.get("label", "")}</div>'
            f'<div class="vl">{c.get("value", "")}</div>'
            f'<div class="sb">{c.get("sub", "")}</div></div>'
        )
    st.markdown(f'<div class="cc-kpis">{html}</div>', unsafe_allow_html=True)


def alert_banner(level: str, title: str, detail: str = "", risk: float = None) -> None:
    """Big status banner. level: 'danger' | 'warn' | 'ok'. risk: 0..1 (optional)."""
    icon = {"danger": "🔥", "warn": "⚠️", "ok": "✅"}.get(level, "ℹ️")
    risk_html = ""
    if risk is not None:
        pct = max(0, min(100, int(round(float(risk) * 100))))
        risk_html = (
            f'<div class="risk"><small>RISK SCORE · {pct}%</small>'
            f'<div class="track"><div class="fill" style="width:{pct}%"></div></div></div>'
        )
    st.markdown(
        f'<div class="cc-alert {level}"><div class="big">{icon}</div>'
        f"<div><h2>{title}</h2><p>{detail}</p></div>{risk_html}</div>",
        unsafe_allow_html=True,
    )


def sidebar_status(title: str, line: str) -> None:
    """Small glass status card for the sidebar."""
    st.markdown(
        f'<div class="cc-status"><b>{title}</b><br><span>{line}</span></div>',
        unsafe_allow_html=True,
    )


# Backwards-compatible alias: older app.py versions import `apply_theme`.
apply_theme = inject_theme