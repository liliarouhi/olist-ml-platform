# Setup
Environment: macOS, Python 3.11 (virtual environment `.venv`), PostgreSQL 18 (Postgres.app), VS Code.

1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -e .`
3. `createdb olist`
4. Download the Olist CSV files into `data/raw/` (see `data/README.md`)
5. `psql olist -f sql/00_staging.sql`
6. `psql olist -f sql/01_schema.sql`
7. `python scripts/load_data.py`
8. `psql olist -f sql/02_indexes.sql`