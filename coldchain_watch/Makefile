# ColdChain Watch Pakistan — Makefile
# =====================================
# The hackathon guide requires a single `make data` command that:
#   1. Fetches raw data from all sources
#   2. Builds Silver (cleaned) layer
#   3. Builds Gold (analysis-ready) layer
#   4. Runs quality checks
#   5. Finishes with NO manual steps
#
# Usage:
#   make data        — full pipeline from scratch
#   make ingest      — Bronze only (fetch raw data)
#   make silver      — Bronze → Silver (clean + validate)
#   make gold        — Silver → Gold (feature engineering)
#   make verify      — re-hash all Bronze files vs manifest
#   make dashboard   — launch the Streamlit dashboard
#   make clean       — remove generated data (keeps Bronze raw files)

.PHONY: data ingest silver gold verify dashboard clean all

# Full pipeline — this is what the jury runs
data: ingest silver gold verify
	@echo ""
	@echo "========================================="
	@echo " ColdChain Watch pipeline complete"
	@echo " Bronze → Silver → Gold"
	@echo " manifest.csv verified"
	@echo "========================================="

# Step 1: Fetch raw data, land in Bronze, record in manifest.csv
ingest:
	@echo "[1/4] Fetching raw data into Bronze..."
	python -m pipeline.ingest

# Step 2: Clean + validate Bronze → Silver
silver:
	@echo "[2/4] Building Silver layer (cleaning + DQ Gate 1)..."
	python -m pipeline.silver

# Step 3: Feature engineering Silver → Gold
gold:
	@echo "[3/4] Building Gold layer (features + DQ Gate 2)..."
	python -m pipeline.gold

# Verify all Bronze files match their manifest hashes
verify:
	@echo "[4/4] Verifying manifest SHA-256 hashes..."
	python -m pipeline.verify_manifest

# Train the Random Forest model on Gold data
model:
	@echo "Training heat exposure model..."
	python -m model.train

# Build the RAG vector index from documents + Gold rows
index:
	@echo "Building RAG index..."
	python -m rag.index

# Run the golden evaluation set on the Insight Agent
eval:
	@echo "Running golden evaluation set..."
	python -m genai.eval

# Launch the Streamlit dashboard locally
dashboard:
	streamlit run dashboard/app.py

# Remove Silver and Gold (keeps Bronze — never delete raw evidence)
clean:
	@echo "Cleaning Silver and Gold layers (Bronze preserved)..."
	python -c "import shutil, os; [shutil.rmtree(d, ignore_errors=True) for d in ['data/silver', 'data/gold']]; print('Done.')"

# Run pipeline once (for testing)
run-once:
	@echo "Running pipeline once..."
	python -m pipeline.scheduler --once

# Start the hourly scheduler (blocks — run in a separate terminal)
run:
	@echo "Starting hourly scheduler (Ctrl+C to stop)..."
	python -m pipeline.scheduler

# Start with Prefect (requires Prefect server running)
run-prefect:
	@echo "Starting Prefect flow..."
	python -m pipeline.scheduler --prefect

# Export prefect_runs.csv for submission
export-runs:
	@echo "Exporting pipeline runs for submission..."
	python -m pipeline.scheduler --export

# Full build including model training
all: data model index
