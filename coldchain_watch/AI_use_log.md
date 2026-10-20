# AI Use Log — ColdChain Watch Pakistan

HackInsight 2026 · Data-to-Decision  
Team: ColdChain Watch

---

## How to read this log

Each entry records one AI-assisted step: the tool used, the task it helped with,
what we kept or changed, and which team member can explain that code.

---

## Entries

### Entry 001
- **Tool and version:** Kiro AI (Claude-based coding assistant), session Oct 19–21 2026
- **Task:** Project scaffolding — directory structure, requirements.txt, Makefile, .env.example
- **What we kept or changed:** Kept the directory layout (bronze/silver/gold pattern). Adjusted paths to match Windows environment. Verified every directory was created before writing files.
- **Who can explain it:** All team members (it is the project skeleton).

---

### Entry 002
- **Tool and version:** Kiro AI
- **Task:** `pipeline/config.py` — central configuration for locations, API endpoints, crop thresholds
- **What we kept or changed:** Kept FAO thresholds (verified against FAO Postharvest Compendium). Added Pakistan city coordinates from Google Maps. Kept API URLs after verifying them against official Open-Meteo and NASA POWER documentation.
- **Who can explain it:** Team lead (pipeline owner).

---

### Entry 003
- **Tool and version:** Kiro AI
- **Task:** `pipeline/ingest.py` — Bronze ingestion from Open-Meteo Historical API and NASA POWER API
- **What we kept or changed:** Kept the API parameter structure after verifying it against each API's official docs. Adjusted timeout values after testing on our machines. Added `time.sleep()` calls to respect free-tier rate limits — AI did not include these initially.
- **Who can explain it:** Team lead (pipeline owner).

---

### Entry 004
- **Tool and version:** Kiro AI
- **Task:** `pipeline/manifest.py` — SHA-256 hashing and manifest.csv recording
- **What we kept or changed:** Kept as generated. Verified the SHA-256 logic independently using Python's hashlib docs. The manifest format matches the guide requirement exactly.
- **Who can explain it:** Team lead, Student 3 (integration).

---

### Entry 005
- **Tool and version:** Kiro AI
- **Task:** `pipeline/silver.py` — DQ Gate 1: Bronze → Silver cleaning
- **What we kept or changed:** The AI proposed the fix/fill/flag decision logic after we explained the guide rules. We verified each rule against the guide and the actual data. Added the NASA -999.0 fill value fix after inspecting the raw JSON and confirming it was needed. Cross-validation threshold (5°C for temperature, 20% for humidity) set by team based on domain knowledge.
- **Who can explain it:** Student 1 (DQ Agent developer).

---

### Entry 006
- **Tool and version:** Kiro AI
- **Task:** `pipeline/gold.py` — DQ Gate 2: Silver → Gold, feature engineering
- **What we kept or changed:** Kept rolling statistics and lag features. Added season map for Pakistan climate seasons (winter/spring/monsoon/autumn) — AI used generic Northern Hemisphere seasons initially; we corrected to match Pakistani agriculture calendar. Humidex formula verified against Environment Canada's original formula. Heat alert flags verified against FAO thresholds.
- **Who can explain it:** Team lead (feature engineering).

---

### Entry 007
- **Tool and version:** Kiro AI
- **Task:** `model/train.py` — Random Forest regressor for temperature forecasting
- **What we kept or changed:** Kept the time-based train/test split logic (critical for guide compliance). Adjusted `N_ESTIMATORS` from 300 to 200 after timing tests — 300 trees was too slow for hourly retraining. Added the baseline comparison logic ourselves to ensure it used `temp_lag_168h` exactly as declared. Bootstrap uncertainty method kept after verifying it matches what the guide calls "prediction intervals".
- **Who can explain it:** Team lead (ML owner).

---

### Entry 008
- **Tool and version:** Kiro AI
- **Task:** `rag/documents.py` — reference document library (FAO, PMD, WFP text chunks)
- **What we kept or changed:** AI drafted the text summaries. Every fact was verified against the original sources before inclusion: FAO Postharvest Compendium for thresholds, PMD bulletins for Karachi climate descriptions, WFP HDX for price patterns. One PMD temperature figure was corrected (AI said 43°C heatwave threshold; PMD definition is 40°C for coastal stations).
- **Who can explain it:** Student 2 (RAG developer).

---

### Entry 009
- **Tool and version:** Kiro AI
- **Task:** `rag/index.py` and `rag/retrieve.py` — TF-IDF vector index and cosine retrieval
- **What we kept or changed:** Originally designed for ChromaDB + sentence-transformers; switched to TF-IDF + sklearn when C++ compiler and torch DLLs were unavailable on team machines. We verified cosine similarity logic is correct (L2-normalised dot product). Top-k values (4 docs, 3 gold rows) chosen by team after testing retrieval quality on sample questions.
- **Who can explain it:** Student 2 (RAG developer).

---

### Entry 010
- **Tool and version:** Kiro AI
- **Task:** `genai/assistant.py` — Gemini-powered Insight Agent with grounding prompt
- **What we kept or changed:** Kept the system prompt structure. Added the explicit "never say currently, always say as of [timestamp]" rule ourselves after re-reading the guide's requirement on `event_time`. The fallback rule-based answer was added by the team to ensure the system works even without a Gemini API key. Golden evaluation set designed by team to match guide's four test types.
- **Who can explain it:** Student 2 (RAG/GenAI developer).

---

### Entry 011
- **Tool and version:** Kiro AI
- **Task:** `dashboard/app.py` — Streamlit dashboard
- **What we kept or changed:** Kept the 5-tab layout. The three mandatory guide panels (freshness, quality status, last run) were placed in the top status bar as required. Dark theme chosen for readability of charts. Crop threshold lines on the temperature chart added by team — AI had them as horizontal annotations but we moved them to `add_hline` for Plotly compatibility.
- **Who can explain it:** Student 3 (dashboard developer).

---

### Entry 012
- **Tool and version:** Kiro AI
- **Task:** `pipeline/scheduler.py` — Prefect hourly flow + fallback schedule runner
- **What we kept or changed:** Kept the step ordering (fetch → silver → gold → retrain → reindex → log). The "retrain only if new data doesn't degrade MAE" logic was added by team — AI had unconditional retraining. The 24-hour minimum interval between retrains was added by team to prevent thrashing on small hourly data increments.
- **Who can explain it:** Student 1 (pipeline operator).

---

## Summary

| Entries | AI-generated and kept as-is | AI-generated and modified | Team-authored |
|---------|----------------------------|--------------------------|---------------|
| 12 | 3 (scaffolding, manifest, config structure) | 8 (all core logic) | 1 (this log) |

All AI-assisted code has been read, understood, and verified by at least one team member.
Every team member can answer viva questions about the code in their area.

---

*Last updated: Oct 2026*
