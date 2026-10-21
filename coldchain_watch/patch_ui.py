import io, os

THEME = r'''"""Black + electric-blue theme for ColdChain Watch. Presentation only."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--g:#2f8bff;--g2:#66b3ff;--bg:#040912;--card:rgba(10,18,34,.72);--line:rgba(47,139,255,.16);--txt:#e8f1ff;--mut:#7d93b5}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif!important}
.stApp{background:radial-gradient(1100px 520px at 12% -8%,rgba(47,139,255,.10),transparent 60%),
 radial-gradient(900px 480px at 100% 0%,rgba(102,179,255,.07),transparent 55%),#040912!important;color:var(--txt)}
header[data-testid="stHeader"]{background:transparent!important}
.block-container{padding-top:2rem!important;max-width:1280px}
h1,h2,h3,h4{color:#fff!important;letter-spacing:-.02em;font-weight:700!important}
hr{border-color:var(--line)!important}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-thumb{background:#15284a;border-radius:8px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#030710,#071022)!important;border-right:1px solid var(--line)}
[data-testid="stSidebar"] h1{font-size:1.35rem!important;background:linear-gradient(90deg,#fff,var(--g));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:var(--mut)!important}
.cc-hero{position:relative;overflow:hidden;padding:30px 34px;margin-bottom:22px;border-radius:22px;
 border:1px solid var(--line);background:linear-gradient(135deg,rgba(47,139,255,.10),rgba(5,8,6,.9) 55%);
 box-shadow:0 0 60px rgba(47,139,255,.07),inset 0 1px 0 rgba(255,255,255,.04)}
.cc-hero:before{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(47,139,255,.05) 1px,transparent 1px),
 linear-gradient(90deg,rgba(47,139,255,.05) 1px,transparent 1px);background-size:34px 34px;
 mask-image:linear-gradient(90deg,transparent,#000 70%);-webkit-mask-image:linear-gradient(90deg,transparent,#000 70%)}
.cc-live{position:relative;display:inline-flex;align-items:center;gap:8px;font:700 .72rem 'JetBrains Mono',monospace;
 letter-spacing:.14em;color:var(--g);padding:6px 14px;border:1px solid var(--line);border-radius:999px;background:rgba(47,139,255,.07)}
.cc-dot{width:8px;height:8px;border-radius:50%;background:var(--g);box-shadow:0 0 10px var(--g);animation:pulse 1.6s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.5)}}
.cc-title{position:relative;margin:14px 0 6px;font-size:2.4rem;font-weight:800;letter-spacing:-.03em;color:#fff;line-height:1.1}
.cc-title span{background:linear-gradient(90deg,var(--g),var(--g2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.cc-sub{position:relative;color:var(--mut);font-size:1rem}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px 20px;
 backdrop-filter:blur(10px);transition:all .25s ease;box-shadow:0 8px 30px rgba(0,0,0,.45)}
[data-testid="stMetric"]:hover{transform:translateY(-4px);border-color:rgba(47,139,255,.55);box-shadow:0 0 28px rgba(47,139,255,.16),0 12px 34px rgba(0,0,0,.5)}
[data-testid="stMetricLabel"] p{color:var(--mut)!important;font-weight:600;text-transform:uppercase;font-size:.72rem!important;letter-spacing:.09em}
[data-testid="stMetricValue"]{color:#fff!important;font-weight:800!important;font-family:'JetBrains Mono',monospace!important}
[data-baseweb="tab-list"],[role="tablist"]{gap:8px;background:rgba(8,14,28,.8);padding:6px;border-radius:999px;border:1px solid var(--line);width:fit-content;max-width:100%;overflow-x:auto}
[role="tab"]{border-radius:999px!important;padding:8px 18px!important;color:var(--mut)!important;font-weight:600!important;transition:all .2s}
[role="tab"]:hover{color:#fff!important;background:rgba(47,139,255,.08)!important}
[role="tab"][aria-selected="true"]{background:linear-gradient(90deg,var(--g),var(--g2))!important;color:#ffffff!important;box-shadow:0 0 20px rgba(47,139,255,.35)}
[role="tab"][aria-selected="true"] p{color:#ffffff!important}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}
[data-baseweb="select"]>div,[data-baseweb="input"],textarea,.stTextInput input{background:#0a1424!important;border:1px solid var(--line)!important;border-radius:12px!important;color:#fff!important}
[data-baseweb="select"]>div:hover,textarea:focus,.stTextInput input:focus{border-color:var(--g)!important;box-shadow:0 0 0 1px var(--g)!important}
[data-testid="stSlider"] [role="slider"]{background:var(--g)!important;box-shadow:0 0 12px var(--g)}
.stButton>button,.stDownloadButton>button,[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-secondary"]{
 background:linear-gradient(90deg,var(--g),var(--g2))!important;color:#ffffff!important;font-weight:700!important;border:0!important;border-radius:12px!important;
 padding:.55rem 1.4rem!important;transition:all .2s;box-shadow:0 0 18px rgba(47,139,255,.25)}
.stButton>button:hover,[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-secondary"]:hover{transform:translateY(-2px);box-shadow:0 0 30px rgba(47,139,255,.5)}
.stButton>button p,[data-testid="stBaseButton-primary"] p,[data-testid="stBaseButton-secondary"] p{color:#ffffff!important}
[data-testid="stExpander"],[data-testid="stAlert"],[data-testid="stDataFrame"]{border:1px solid var(--line)!important;border-radius:14px!important;background:var(--card)!important}
[data-testid="stPlotlyChart"]{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:10px}
code{color:var(--g)!important;background:rgba(47,139,255,.08)!important;border-radius:6px}
a{color:var(--g)!important}
footer{visibility:hidden}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(city: str):
    st.markdown(
        f"""<div class="cc-hero"><div class="cc-live"><span class="cc-dot"></span>LIVE MONITORING</div>
        <div class="cc-title">ColdChain <span>Watch</span> Pakistan</div>
        <div class="cc-sub">Live heat exposure &amp; price outlook for cold-store managers · {city}</div></div>""",
        unsafe_allow_html=True,
    )
'''

def rd(p):
    return io.open(p, encoding="utf-8").read()
def wr(p, s):
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)

wr("dashboard/theme.py", THEME)

s = rd("dashboard/app.py")
if "apply_theme" in s:
    print("app.py already patched"); raise SystemExit
s = s.replace("    sys.path.insert(0, BASE_DIR)\n",
              "    sys.path.insert(0, BASE_DIR)\n\nfrom dashboard.theme import apply_theme, hero  # noqa: E402\n", 1)
s = s.replace("# ─── Paths ─", "apply_theme()\n\n# ─── Paths ─", 1)
old_hero = ('st.markdown("## ❄️ ColdChain Watch Pakistan")\n'
            'st.markdown(f"*Live heat exposure & price outlook for cold-store managers · {city}*")\n'
            'st.divider()')
assert old_hero in s, "hero text not found"
s = s.replace(old_hero, "hero(city)")
for a, b in {'"#0f172a"': '"rgba(0,0,0,0)"', '"#1e293b"': '"#0f1d36"', '"#e2e8f0"': '"#dbe8ff"',
             '"#3b82f6"': '"#2f8bff"', '"#93c5fd"': '"#9cc9ff"', '"#06b6d4"': '"#66b3ff"',
             '"#94a3b8"': '"#3b4f73"'}.items():
    s = s.replace(a, b)
wr("dashboard/app.py", s)

c = rd(".streamlit/config.toml")
c = (c.replace("#16a34a", "#2f8bff").replace("#0f172a", "#040912")
      .replace("#1e293b", "#0a1424").replace("#e2e8f0", "#e8f1ff"))
wr(".streamlit/config.toml", c)
print("DONE - UI patched")