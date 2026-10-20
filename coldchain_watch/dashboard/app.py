"""
dashboard/app.py
----------------
ColdChain Watch Pakistan — Streamlit Dashboard

Guide requirements (mandatory for submission):
  1. Data freshness   — when each source was last fetched, whether it is late
  2. Quality status   — latest result of the Data Quality Agent
  3. Last successful run — time and status of the latest pipeline run

Plus our value-add panels:
  - Live temperature chart (last 7 days)
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


# ─── Data loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_gold() -> pd.DataFrame:
    if not os.path.exists(GOLD_PATH):
        return pd.DataFrame()
    df = pd.read_parquet(GOLD_PATH)
    df = df.sort_values("timestamp_utc")
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


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://flagcdn.com/w40/pk.png", width=40)
    st.title("ColdChain Watch 🇵🇰")
    st.markdown("**Decision support for cold-store managers in Pakistan**")
    st.divider()

    city     = st.selectbox("City", ["Karachi"], index=0)
    crop     = st.selectbox("Crop", ["potato", "tomato", "onion"], index=0)
    days_back = st.slider("History (days)", min_value=7, max_value=60, value=14)

    st.divider()
    st.caption("Data sources")
    st.caption("📡 Open-Meteo API (CC BY 4.0)")
    st.caption("🛰️ NASA POWER API (NASA Open Data)")
    st.caption("📊 WFP HDX Food Prices")
    st.caption("📋 FAO Postharvest Guidelines")
    st.divider()
    st.caption("HackInsight 2026 · Team ColdChain")


# ─── Load data ──────────────────────────────────────────────────────────────

df       = load_gold()
manifest = load_manifest()
metrics  = load_metrics()

if df.empty:
    st.error("Gold data not found. Run `make data` first.")
    st.stop()

# Filter to selected window
df_window = df.tail(days_back * 24).copy()
df_recent = df.tail(168).copy()   # last 7 days always


# ─── SECTION 1: Status bar (mandatory guide requirements) ───────────────────

st.markdown("## ❄️ ColdChain Watch Pakistan")
st.markdown(f"*Live heat exposure & price outlook for cold-store managers · {city}*")
st.divider()

col1, col2, col3, col4 = st.columns(4)

# 1a. Data freshness
latest_ts    = df["timestamp_utc"].max()
now_utc      = pd.Timestamp.now(tz="UTC")
age_hours    = (now_utc - latest_ts).total_seconds() / 3600
freshness_ok = age_hours < 26

with col1:
    st.metric(
        label="📡 Data as of (PKT)",
        value=str(df["timestamp_pkt"].max())[:16],
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
        delta=f"Risk score: {risk_val:.2f} | Temp: {current['temp_c']:.1f}°C",
        delta_color="inverse" if alert_val == 1 else "normal",
    )

st.divider()


# ─── SECTION 2: Temperature chart ───────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡️ Temperature", "🚨 Heat Alerts", "📊 Model Performance",
    "💬 Insight Agent", "📋 Pipeline Status"
])

with tab1:
    st.subheader(f"Temperature — {city} (last {days_back} days)")

    fig = go.Figure()

    # Actual temperature
    fig.add_trace(go.Scatter(
        x=df_window["timestamp_pkt"].astype(str),
        y=df_window["temp_c"],
        name="Temperature (°C)",
        line=dict(color="#3b82f6", width=1.5),
        hovertemplate="%{x}<br>Temp: %{y:.1f}°C<extra></extra>",
    ))

    # Feels-like
    fig.add_trace(go.Scatter(
        x=df_window["timestamp_pkt"].astype(str),
        y=df_window["feels_like_c"],
        name="Feels Like (°C)",
        line=dict(color="#93c5fd", width=1, dash="dot"),
        hovertemplate="%{x}<br>Feels like: %{y:.1f}°C<extra></extra>",
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
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Humidity
    st.subheader("Relative Humidity")
    fig2 = px.area(
        df_window,
        x=df_window["timestamp_pkt"].astype(str),
        y="humidity_pct",
        labels={"humidity_pct": "Humidity (%)", "x": "Time (PKT)"},
        color_discrete_sequence=["#06b6d4"],
        height=250,
    )
    fig2.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─── SECTION 3: Heat alerts ──────────────────────────────────────────────────

with tab2:
    st.subheader(f"Heat Alert Timeline — last {days_back} days")

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
        x=df_window["timestamp_pkt"].astype(str),
        y=f"heat_risk_{crop}",
        labels={f"heat_risk_{crop}": "Heat Risk Score", "x": "Time (PKT)"},
        color_discrete_sequence=[CROP_COLORS[crop]],
        range_y=[0, 1],
        height=300,
    )
    fig3.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Season breakdown
    st.subheader("Temperature by Season")
    season_stats = df.groupby("season")["temp_c"].agg(["mean", "min", "max", "count"]).round(2)
    season_stats.columns = ["Mean (°C)", "Min (°C)", "Max (°C)", "Hours"]
    st.dataframe(season_stats, use_container_width=True)


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
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        # Bar chart
        fig4 = go.Figure()
        horizons = [m["horizon"] for m in metrics]
        fig4.add_trace(go.Bar(
            name="Model MAE",
            x=horizons,
            y=[m["mae_model"] for m in metrics],
            marker_color="#3b82f6",
        ))
        fig4.add_trace(go.Bar(
            name="Baseline MAE",
            x=horizons,
            y=[m["mae_baseline"] for m in metrics],
            marker_color="#94a3b8",
        ))
        fig4.update_layout(
            barmode="group",
            title="MAE: Model vs Baseline (lower is better)",
            yaxis_title="MAE (°C)",
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
            height=350,
        )
        st.plotly_chart(fig4, use_container_width=True)
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
            color_continuous_scale="Blues",
            height=400,
        )
        fig5.update_layout(
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Errors by group
    grp_path = os.path.join(BASE_DIR, "model", "artifacts", "errors_by_group_6h.csv")
    if os.path.exists(grp_path):
        st.subheader("Errors by Group (6h Forecast)")
        grp_df = pd.read_csv(grp_path)
        st.dataframe(grp_df, use_container_width=True, hide_index=True)

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
    st.subheader("💬 ColdChain Insight Agent")
    st.markdown(
        "Ask a question about heat exposure, storage, or dispatch decisions. "
        "Every answer cites its source. Numbers come from Gold tables only."
    )

    # Show what the agent can answer
    with st.expander("📖 Example questions"):
        st.markdown("""
- *What is the current temperature in Karachi and should I pre-cool my potato store?*
- *When is the best time to dispatch tomato trucks today?*
- *What humidity level should I maintain for onion storage during monsoon?*
- *Should I hold or sell my tomato stock this week?*
- *How many kilograms of potatoes spoiled today?* ← agent will say "I don't know"
        """)

    question = st.text_input(
        "Your question:",
        placeholder="e.g. Should I dispatch tomato trucks tonight?",
        key="agent_question",
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving evidence and generating answer..."):
            try:
                from genai.assistant import InsightAgent
                agent = InsightAgent()
                answer = agent.ask(question)

                if answer.is_i_dont_know:
                    st.warning(f"**{answer.answer}**")
                else:
                    st.success(answer.answer)

                st.markdown("**Sources cited:**")
                for cite in answer.citations:
                    st.markdown(f"- `{cite}`")

                col_a, col_b = st.columns(2)
                col_a.metric("Evidence found", "✅ Yes" if answer.has_evidence else "❌ No")
                col_b.metric("Model used", answer.model_used)

                if answer.model_used == "fallback":
                    st.info(
                        "💡 Full AI answers require a Gemini API key. "
                        "Set `GEMINI_API_KEY` in `.env` to enable. "
                        "Get a free key at https://aistudio.google.com"
                    )

            except Exception as e:
                st.error(f"Agent error: {e}")

    # Golden eval results if available
    eval_path = os.path.join(BASE_DIR, "model", "artifacts", "golden_eval.json")
    if os.path.exists(eval_path):
        st.divider()
        st.subheader("🎯 Golden Evaluation Results")
        with open(eval_path) as f:
            eval_results = json.load(f)
        eval_df = pd.DataFrame(eval_results)
        eval_df["passed"] = eval_df["passed"].map({True: "✅ Pass", False: "❌ Fail", None: "⚠️ N/A"})
        st.dataframe(
            eval_df[["eval_id", "question", "expected", "passed", "n_citations", "model_used"]],
            use_container_width=True,
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
            use_container_width=True,
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
    st.dataframe(df[summary_cols].describe().round(3), use_container_width=True)

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
