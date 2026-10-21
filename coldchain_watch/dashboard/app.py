"""
dashboard/app.py
----------------
ColdChain Watch Pakistan — Streamlit Dashboard

Guide requirements (mandatory for submission):
  1. Data freshness   — when each source was last fetched, whether it is late
  2. Quality status   — latest result of the Data Quality Agent
  3. Last successful run — time and status of the latest pipeline run

Plus our value-add panels:
  - Live temperature chart (selectable history window)
  - Heat alert timeline per crop
  - Model performance vs baseline
  - RAG Insight Agent chat interface
  - Pakistan Impact section

Run locally:  streamlit run dashboard/app.py
Deploy:       Push to Hugging Face Space (same command)
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import inject_theme, hero, window_badge  # presentation only

# ─── Path setup ─────────────────────────────────────────────────────────────
# Streamlit Cloud runs from repo root. Add coldchain_watch/ to sys.path
# so that `from genai.assistant import ...` resolves correctly.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ColdChain Watch Pakistan",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

# ─── Font: Helvetica across the whole app ───────────────────────────────────
# Helvetica ships with macOS/iOS; Windows/Linux fall back to Arial, which looks
# nearly identical. We avoid a blanket `* {}` rule because it would break
# Streamlit's Material icon font (arrows would render as text).
FONT_FAMILY = 'Helvetica, "Helvetica Neue", Arial, sans-serif'

st.markdown(
    f"""
    <style>
    html, body, .stApp, .stMarkdown, .stText, .stCaption, .stButton button,
    .stTextInput input, .stTextArea textarea, .stSelectbox, .stMultiSelect,
    .stSlider, .stDataFrame, .stTable, .stMetric, .stTabs, .stRadio,
    .stCheckbox, .stAlert, .stExpander, .stSpinner,
    [data-testid="stSidebar"], [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"],
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
    button[role="tab"],
    h1, h2, h3, h4, h5, h6, p, label, li, td, th, code, pre, input, textarea {{
        font-family: {FONT_FAMILY} !important;
    }}
    [data-testid="stChatMessage"], [data-testid="stChatInput"] textarea {{
        font-family: {FONT_FAMILY} !important;
    }}
    /* Metric cards: wrap long text instead of truncating it */
    [data-testid="stMetric"] {{ min-height: 170px !important; height: 100% !important; box-sizing: border-box !important; overflow: hidden !important; }}
    [data-testid="stMetric"] * {{ max-width: 100% !important; }}
    [data-testid="stColumn"] > div, [data-testid="stColumn"] [data-testid="stMetric"] {{ width: 100% !important; }}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
        line-height: 1.15 !important;
        font-size: clamp(1rem, 1.35vw, 1.35rem) !important;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }}
    [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.3 !important;
        height: auto !important;
        max-width: 100% !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig):
    """Apply Helvetica to a Plotly figure (charts don't inherit page CSS)."""
    fig.update_layout(font=dict(family=FONT_FAMILY))
    return fig


# ─── Paths ──────────────────────────────────────────────────────────────────
GOLD_PATH     = os.path.join(BASE_DIR, "data", "gold", "weather_gold_karachi.parquet")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "manifest.csv")
METRICS_PATH  = os.path.join(BASE_DIR, "model", "artifacts", "metrics_summary.json")
DQ_LOG_DIR    = os.path.join(BASE_DIR, "data", "logs")
MODEL_CARD    = os.path.join(BASE_DIR, "model", "model_card.md")

CROP_COLORS = {
    "potato": "#f59e0b",
    "tomato": "#ef4444",
    "onion":  "#8b5cf6",
}

CROP_THRESHOLDS = {
    "potato": 4.0,
    "tomato": 13.0,
    "onion":  2.0,
}

MIN_HISTORY_DAYS = 7
MAX_HISTORY_DAYS = 60
DEFAULT_HISTORY_DAYS = 14


# ─── Data loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_gold() -> pd.DataFrame:
    if not os.path.exists(GOLD_PATH):
        return pd.DataFrame()
    df = pd.read_parquet(GOLD_PATH)

    # Real datetimes so the History slider can filter by DATE (not by row count)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    df = df.reset_index(drop=True)

    # Local-time column used on chart x-axes (a real datetime axis, not text labels)
    ts = pd.to_datetime(df["timestamp_pkt"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    df["ts_plot"] = ts
    return df


@st.cache_data(ttl=300)
def load_manifest() -> list[dict]:
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@st.cache_data(ttl=300)
def load_metrics() -> list[dict]:
    if not os.path.exists(METRICS_PATH):
        return []
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_latest_dq_events() -> list[str]:
    """Read the last 20 lines from the most recent DQ gate log."""
    events = []
    for log_name in ["dq_gate2", "dq_gate1"]:
        for fname in sorted(os.listdir(DQ_LOG_DIR), reverse=True):
            if fname.startswith(log_name):
                path = os.path.join(DQ_LOG_DIR, fname)
                try:
                    with open(path, encoding="utf-8") as f:
                        lines = f.readlines()
                    events = [l.strip() for l in lines[-20:] if l.strip()]
                    return events
                except Exception:
                    pass
    return events


def window_by_days(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return only the rows from the last `days` days, counted back from the newest reading."""
    end = frame["timestamp_utc"].max()
    start = end - pd.Timedelta(days=days)
    return frame[frame["timestamp_utc"] > start].copy()


# ─── Load data (before the sidebar so the slider knows how much history exists) ──

df       = load_gold()
manifest = load_manifest()
metrics  = load_metrics()

if not df.empty:
    available_days = int((df["timestamp_utc"].max() - df["timestamp_utc"].min()).total_seconds() // 86400)
else:
    available_days = 0


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://flagcdn.com/w40/pk.png", width=40)
    st.title("ColdChain Watch 🇵🇰")
    st.markdown("**Decision support for cold-store managers in Pakistan**")
    st.divider()

    city = st.selectbox("City", ["Karachi"], index=0)
    crop = st.selectbox("Crop", ["potato", "tomato", "onion"], index=0)

    # History slider: limited to the days of data that really exist
    slider_max = max(1, min(MAX_HISTORY_DAYS, available_days))
    slider_min = min(MIN_HISTORY_DAYS, slider_max)
    if slider_max > slider_min:
        days_back = st.slider(
            "History (days)",
            min_value=slider_min,
            max_value=slider_max,
            value=min(DEFAULT_HISTORY_DAYS, slider_max),
            step=1,
            key="history_days",
            help="How many days back (from the newest reading) the charts and alert "
                 "percentages should cover.",
        )
    else:
        days_back = slider_max
        st.caption(f"History: {days_back} day(s) of data available")

    st.divider()
    st.caption("Data sources")
    st.caption("📡 Open-Meteo API (CC BY 4.0)")
    st.caption("🛰️ NASA POWER API (NASA Open Data)")
    st.caption("📊 WFP HDX Food Prices")
    st.caption("📋 FAO Postharvest Guidelines")
    st.divider()
    st.caption("HackInsight 2026 · Team ColdChain")


if df.empty:
    st.error("Gold data not found. Run `make data` first.")
    st.stop()

# Filter to the selected window (by date, counted back from the newest reading)
df_window = window_by_days(df, days_back)
window_start = df_window["ts_plot"].min()
window_end   = df_window["ts_plot"].max()
window_text  = (
    f"📅 Showing last {days_back} days · "
    f"{window_start:%d %b %Y} → {window_end:%d %b %Y} · {len(df_window):,} readings"
)


# ─── SECTION 1: Status bar (mandatory guide requirements) ───────────────────

hero(city)

col1, col2, col3, col4 = st.columns(4)

# 1a. Data freshness
latest_ts    = df["timestamp_utc"].max()
now_utc      = pd.Timestamp.now(tz="UTC")
age_hours    = (now_utc - latest_ts).total_seconds() / 3600
freshness_ok = age_hours < 26

with col1:
    st.metric(
        label="📡 Data as of (PKT)",
        value=f"{df['ts_plot'].max():%d %b, %H:%M}",
        delta=f"{age_hours:.1f}h ago",
        delta_color="normal" if freshness_ok else "inverse",
    )

# 1b. DQ gate status
dq_events  = get_latest_dq_events()
gate_passed = any("GATE2 OK" in e for e in dq_events)
gate_failed = any("GATE FAIL" in e or "GATE2 FAIL" in e for e in dq_events)

with col2:
    if gate_failed:
        st.metric("🔴 DQ Gate 2", "FAILED", delta="Check logs")
    elif gate_passed:
        st.metric("🟢 DQ Gate 2", "PASS", delta="All checks OK")
    else:
        st.metric("⚪ DQ Gate 2", "Unknown", delta="No recent run")

# 1c. Model status
beats_all = all(m["beats_baseline"] for m in metrics) if metrics else False
with col3:
    if metrics:
        best = min(metrics, key=lambda x: x["mae_model"])
        st.metric(
            "🤖 Model MAE (1h)",
            f"{metrics[0]['mae_model']:.3f}°C",
            delta=f"vs baseline {metrics[0]['mae_baseline']:.2f}°C ✅" if beats_all else "❌ fails baseline",
            delta_color="normal" if beats_all else "inverse",
        )
    else:
        st.metric("🤖 Model", "Not trained", delta="Run `make model`")

# 1d. Current heat alert
current = df.iloc[-1]
alert_val = int(current[f"heat_alert_{crop}"])
risk_val  = float(current[f"heat_risk_{crop}"])
with col4:
    alert_label = "🔴 HEAT ALERT" if alert_val == 1 else "🟢 OK"
    st.metric(
        f"🌡️ {crop.title()} Alert Now",
        alert_label,
        delta=f"Risk {risk_val:.2f} · {current['temp_c']:.1f}°C",
        delta_color="inverse" if alert_val == 1 else "normal",
    )

st.divider()


# ─── SECTION 2: Temperature chart ───────────────────────────────────────────
# NOTE: charts deliberately do NOT set font_color / grid colours, so Plotly
# follows Streamlit's Light / Dark theme automatically.
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Heat Alerts",
    "Model",
    "AI Insights",
    "Pipeline",
])

with tab1:
    st.subheader(f"Temperature — {city} (last {days_back} days)")
    window_badge(window_text)

    fig = go.Figure()

    # Actual temperature
    fig.add_trace(go.Scatter(
        x=df_window["ts_plot"],
        y=df_window["temp_c"],
        name="Temperature (°C)",
        line=dict(color="#4f5df0", width=2),
        hovertemplate="%{x|%d %b %H:%M}<br>Temp: %{y:.1f}°C<extra></extra>",
    ))

    # Feels-like
    fig.add_trace(go.Scatter(
        x=df_window["ts_plot"],
        y=df_window["feels_like_c"],
        name="Feels Like (°C)",
        line=dict(color="#12b3bd", width=1.5, dash="dot"),
        hovertemplate="%{x|%d %b %H:%M}<br>Feels like: %{y:.1f}°C<extra></extra>",
    ))

    # FAO threshold line for selected crop
    threshold = CROP_THRESHOLDS[crop]
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=CROP_COLORS[crop],
        annotation_text=f"FAO {crop.title()} threshold ({threshold}°C)",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title="Time (PKT)",
        yaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        xaxis=dict(range=[window_start, window_end]),
    )
    st.plotly_chart(style_fig(fig), width="stretch")

    # Humidity
    st.subheader("Relative Humidity")
    fig2 = px.area(
        df_window,
        x="ts_plot",
        y="humidity_pct",
        labels={"humidity_pct": "Humidity (%)", "ts_plot": "Time (PKT)"},
        color_discrete_sequence=["#12b3bd"],
        height=250,
    )
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[window_start, window_end]),
    )
    st.plotly_chart(style_fig(fig2), width="stretch")


# ─── SECTION 3: Heat alerts ──────────────────────────────────────────────────

with tab2:
    st.subheader(f"Heat Alert Timeline — last {days_back} days")
    window_badge(window_text)

    # Heat alert % by crop
    alert_pcts = {}
    for c in ["potato", "tomato", "onion"]:
        pct = df_window[f"heat_alert_{c}"].mean() * 100
        alert_pcts[c] = round(pct, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("🥔 Potato alerts", f"{alert_pcts['potato']:.1f}%",
              delta="Always above 4°C in Karachi", delta_color="off")
    c2.metric("🍅 Tomato alerts", f"{alert_pcts['tomato']:.1f}%",
              delta="Above 13°C threshold", delta_color="inverse" if alert_pcts["tomato"] > 50 else "normal")
    c3.metric("🧅 Onion alerts", f"{alert_pcts['onion']:.1f}%",
              delta="Always above 2°C in Karachi", delta_color="off")

    st.info(
        "ℹ️ In Karachi, external temperature always exceeds potato and onion FAO thresholds. "
        "These alerts indicate active refrigeration energy cost, not imminent spoilage. "
        "Tomato is the most operationally useful alert.",
        icon="ℹ️"
    )

    # Heat risk score chart for selected crop
    st.subheader(f"Heat Risk Score — {crop.title()} (0=safe, 1=extreme)")
    fig3 = px.area(
        df_window,
        x="ts_plot",
        y=f"heat_risk_{crop}",
        labels={f"heat_risk_{crop}": "Heat Risk Score", "ts_plot": "Time (PKT)"},
        color_discrete_sequence=[CROP_COLORS[crop]],
        range_y=[0, 1],
        height=300,
    )
    fig3.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[window_start, window_end]),
    )
    st.plotly_chart(style_fig(fig3), width="stretch")

    # Season breakdown
    st.subheader("Temperature by Season")
    season_stats = df.groupby("season")["temp_c"].agg(["mean", "min", "max", "count"]).round(2)
    season_stats.columns = ["Mean (°C)", "Min (°C)", "Max (°C)", "Hours"]
    st.dataframe(season_stats, width="stretch")


# ─── SECTION 4: Model performance ───────────────────────────────────────────

with tab3:
    st.subheader("Model Performance vs Declared Baseline")
    st.markdown(
        "**Declared baseline:** Same hour last week (`temp_lag_168h`). "
        "Model must beat this to be accepted."
    )

    if metrics:
        rows = []
        for m in metrics:
            rows.append({
                "Horizon":        m["horizon"],
                "Model MAE (°C)": m["mae_model"],
                "Baseline MAE (°C)": m["mae_baseline"],
                "Improvement":    f"{m['improvement_mae_pct']:+.1f}%",
                "R²":             m["r2_model"],
                "Beats Baseline": "✅ Yes" if m["beats_baseline"] else "❌ No",
            })
        perf_df = pd.DataFrame(rows)
        st.dataframe(perf_df, width="stretch", hide_index=True)

        # Bar chart
        fig4 = go.Figure()
        horizons = [m["horizon"] for m in metrics]
        fig4.add_trace(go.Bar(
            name="Model MAE",
            x=horizons,
            y=[m["mae_model"] for m in metrics],
            marker_color="#4f5df0",
        ))
        fig4.add_trace(go.Bar(
            name="Baseline MAE",
            x=horizons,
            y=[m["mae_baseline"] for m in metrics],
            marker_color="#f2709f",
        ))
        fig4.update_layout(
            barmode="group",
            title="MAE: Model vs Baseline (lower is better)",
            yaxis_title="MAE (°C)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(style_fig(fig4), width="stretch")
    else:
        st.warning("Model not trained yet. Run `make model`.")

    # Feature importance
    fi_path = os.path.join(BASE_DIR, "model", "artifacts", "feature_importance_6h.csv")
    if os.path.exists(fi_path):
        st.subheader("Top 15 Features (6h Forecast)")
        fi_df = pd.read_csv(fi_path).head(15)
        fig5 = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#cfd6ff", "#4f5df0"],
            height=400,
        )
        fig5.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(style_fig(fig5), width="stretch")

    # Errors by group
    grp_path = os.path.join(BASE_DIR, "model", "artifacts", "errors_by_group_6h.csv")
    if os.path.exists(grp_path):
        st.subheader("Errors by Group (6h Forecast)")
        grp_df = pd.read_csv(grp_path)
        st.dataframe(grp_df, width="stretch", hide_index=True)

        worst = grp_df.loc[grp_df["mae"].idxmax()]
        st.warning(
            f"⚠️ Weakest group: **{worst['group_type']}={worst['group_value']}** "
            f"— MAE {worst['mae']:.3f}°C. "
            "Users in this group should apply extra caution to forecast-based decisions."
        )

    if os.path.exists(MODEL_CARD):
        with st.expander("📄 Model Card"):
            with open(MODEL_CARD, encoding="utf-8") as f:
                st.markdown(f.read())


# ─── SECTION 5: Insight Agent ───────────────────────────────────────────────

with tab4:
    head_l, head_r = st.columns([6, 1])
    head_l.subheader("💬 ColdChain Insight Agent")
    head_l.caption(
        "Ask about heat exposure, storage or dispatch decisions. "
        "Every answer cites its sources, and numbers come from Gold tables only."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if head_r.button("🗑️ Clear", width="stretch", help="Start a new conversation"):
        st.session_state.chat_history = []
        st.rerun()

    import random

    QUESTION_BANK = {
        "🌡️ Heat & temperature": [
            "What is the current temperature in Karachi and should I pre-cool my potato store?",
            "How hot will it get in the next 6 hours?",
            "Is there a heat alert for tomatoes right now?",
            "Which months are the hottest in Karachi and why does that matter for cold storage?",
            "How much higher is the feels-like temperature than the actual temperature today?",
            "Why do potato and onion alerts almost always show in Karachi?",
        ],
        "🥔 Storage": [
            "What humidity level should I maintain for onion storage during monsoon?",
            "What is the ideal storage temperature for potatoes?",
            "How do I store tomatoes to extend their shelf life in summer?",
            "What happens to potatoes if they are stored below 4°C?",
            "How can I reduce condensation and rot in an onion store?",
            "How often should I check ventilation in a humid season?",
        ],
        "🚚 Dispatch & timing": [
            "When is the best time to dispatch tomato trucks today?",
            "Should I dispatch tomato trucks tonight?",
            "What time of day is the coolest for loading potatoes?",
            "How should I protect produce during a long truck journey in summer?",
            "Is it safe to move onions to market during a heatwave?",
        ],
        "💰 Market decisions": [
            "Should I hold or sell my tomato stock this week?",
            "How do heat spikes affect potato and tomato prices in Pakistan?",
            "When does it make sense to pay for extra refrigeration?",
            "What does the WFP food price data say about recent price trends?",
        ],
        "🤖 Forecast & data": [
            "How accurate is the temperature forecast model?",
            "What data sources does this dashboard use?",
            "How fresh is the data right now?",
            "Which conditions make the forecast least reliable?",
            "How many kilograms of potatoes spoiled today?",
        ],
    }
    ALL_QUESTIONS = [q for qs in QUESTION_BANK.values() for q in qs]

    def get_agent():
        """Create the Insight Agent once per session."""
        if "insight_agent" not in st.session_state:
            from genai.assistant import InsightAgent
            st.session_state.insight_agent = InsightAgent()
        return st.session_state.insight_agent

    def render_assistant(msg: dict):
        """Draw one assistant answer (text, sources, small footer)."""
        if msg.get("is_idk"):
            st.warning(msg["content"])
        else:
            st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📚 Sources ({len(msg['citations'])})"):
                for cite in msg["citations"]:
                    st.markdown(f"- `{cite}`")
        if msg.get("model_used"):
            evidence = "evidence found" if msg.get("has_evidence") else "no evidence found"
            st.caption(f"Model: {msg['model_used']} · {evidence}")
        if msg.get("error"):
            st.caption(f"⚠️ {msg['error']}")

    # One bordered, scrollable container holds the whole conversation
    chat_box = st.container(height=520, border=True)

    # The input sits directly under the conversation (like ChatGPT)
    typed = st.chat_input("Ask about heat exposure, storage or dispatch…")
    prompt = typed or st.session_state.pop("pending_q", None)

    with chat_box:
        # Empty state: welcome message + a random pick of example questions
        if not st.session_state.chat_history and not prompt:
            with st.chat_message("assistant", avatar="❄️"):
                st.markdown(
                    "Hi! I'm the **ColdChain Insight Agent**. Ask me about heat risk, "
                    "storage conditions, dispatch timing or the forecast, "
                    "or try one of these:"
                )
            seed = st.session_state.get("sugg_seed", 0)
            picks = random.Random(seed).sample(ALL_QUESTIONS, 5)
            for i, q in enumerate(picks):
                if st.button(q, key=f"sugg_{seed}_{i}", width="stretch"):
                    st.session_state.pending_q = q
                    st.rerun()
            if st.button("🔄 Show different questions", key="sugg_shuffle"):
                st.session_state.sugg_seed = seed + 1
                st.rerun()

        # Previous messages
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="❄️"):
                    render_assistant(msg)

        # New question
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant", avatar="❄️"):
                with st.spinner("Retrieving evidence and thinking…"):
                    try:
                        agent = get_agent()
                        answer = agent.ask(prompt)
                        reply = {
                            "role": "assistant",
                            "content": answer.answer,
                            "citations": list(answer.citations or []),
                            "is_idk": bool(answer.is_i_dont_know),
                            "has_evidence": bool(answer.has_evidence),
                            "model_used": answer.model_used,
                            "error": (
                                str(getattr(agent, "_last_error", "") or "")
                                if answer.model_used == "fallback" else ""
                            ),
                        }
                    except Exception as e:
                        reply = {
                            "role": "assistant",
                            "content": f"Sorry, something went wrong: {e}",
                            "is_idk": True,
                        }
                render_assistant(reply)
            st.session_state.chat_history.append(reply)

    with st.expander("💡 Browse example questions"):
        cat_tabs = st.tabs(list(QUESTION_BANK.keys()))
        for ci, (cat, qs) in enumerate(QUESTION_BANK.items()):
            with cat_tabs[ci]:
                for qi, q in enumerate(qs):
                    if st.button(q, key=f"bank_{ci}_{qi}", width="stretch"):
                        st.session_state.pending_q = q
                        st.rerun()

    # Golden eval results if available
    eval_path = os.path.join(BASE_DIR, "model", "artifacts", "golden_eval.json")
    if os.path.exists(eval_path):
        with st.expander("🎯 Golden Evaluation Results"):
            with open(eval_path) as f:
                eval_results = json.load(f)
            eval_df = pd.DataFrame(eval_results)
            eval_df["passed"] = eval_df["passed"].map({True: "✅ Pass", False: "❌ Fail", None: "⚠️ N/A"})
            st.dataframe(
                eval_df[["eval_id", "question", "expected", "passed", "n_citations", "model_used"]],
                width="stretch",
                hide_index=True,
            )


# ─── SECTION 6: Pipeline status ─────────────────────────────────────────────

with tab5:
    st.subheader("📋 Pipeline & Data Quality Status")

    # Manifest
    st.markdown("### Data Manifest (Bronze Sources)")
    if manifest:
        mdf = pd.DataFrame(manifest)
        st.dataframe(
            mdf[["source_name", "fetch_time_utc", "licence", "notes"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("Manifest not found.")

    # DQ events
    st.markdown("### Recent DQ Gate Events")
    dq_events = get_latest_dq_events()
    if dq_events:
        for event in dq_events[-15:]:
            if "FAIL" in event or "FLAG" in event:
                st.markdown(f"🟡 `{event}`")
            elif "OK" in event or "SUCCESS" in event:
                st.markdown(f"🟢 `{event}`")
            else:
                st.markdown(f"⚪ `{event}`")
    else:
        st.info("No DQ events found.")

    # Gold summary
    st.markdown("### Gold Table Summary")
    summary_cols = ["temp_c", "humidity_pct", "heat_alert_potato", "heat_alert_tomato", "heat_alert_onion"]
    st.dataframe(df[summary_cols].describe().round(3), width="stretch")

    # Flagged rows info
    flagged_pct = df["is_flagged"].mean() * 100
    filled_pct  = df["is_filled"].mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", f"{len(df):,}")
    c2.metric("Flagged rows", f"{flagged_pct:.1f}%", delta="Cross-validation flags", delta_color="off")
    c3.metric("Filled rows", f"{filled_pct:.1f}%", delta="Short-gap interpolation", delta_color="off")

    st.markdown("### Log Files")
    log_files = sorted(os.listdir(DQ_LOG_DIR), reverse=True)
    for lf in log_files[:8]:
        lpath = os.path.join(DQ_LOG_DIR, lf)
        size  = os.path.getsize(lpath)
        st.markdown(f"- `{lf}` ({size:,} bytes)")

    st.divider()
    st.markdown("### About This System")
    st.markdown("""
**ColdChain Watch Pakistan** is an end-to-end data pipeline for cold-store managers.

| Component | Status |
|-----------|--------|
| Bronze ingestion (Open-Meteo + NASA POWER) | ✅ Running |
| DQ Gate 1: Bronze → Silver | ✅ Complete |
| DQ Gate 2: Silver → Gold | ✅ Complete |
| RF Temperature Forecast Model | ✅ Trained |
| RAG Index (TF-IDF) | ✅ Built |
| Insight Agent | ✅ Active |
| Prefect Hourly Scheduler | ⏳ Pending |

**Data sources:**
- Open-Meteo Historical API (CC BY 4.0)
- NASA POWER Hourly API (NASA Open Data)
- FAO Postharvest Compendium (Reference)
- WFP HDX Food Prices Pakistan

**HackInsight 2026 · Data-to-Decision · Team ColdChain**
    """)