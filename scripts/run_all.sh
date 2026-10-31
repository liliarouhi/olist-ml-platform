#!/usr/bin/env bash
# Rebuild the whole project from the raw CSV files in data/raw/. Requires a running PostgreSQL.
set -euo pipefail

createdb olist 2>/dev/null || true
PSQL="psql olist -v ON_ERROR_STOP=1 -q -f"

$PSQL sql/00_staging.sql
$PSQL sql/01_schema.sql
python scripts/load_data.py
$PSQL sql/02_indexes.sql
$PSQL sql/04_star_schema.sql
$PSQL sql/05_materialized_views.sql
$PSQL sql/06_features.sql
python scripts/eda.py
python scripts/train_final.py
pytest -q

echo "Done. Dashboard: streamlit run dashboard/app.py | API: uvicorn olist_ml.api.main:app --reload"
