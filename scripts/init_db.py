"""Initialise the SQLite reliability database.

Executes db/schema.sql against the path configured in configs/config.yaml.
Idempotent: re-running when the database already exists is safe — the script
checks for existing tables and skips schema execution if they are present.

Usage:
    python scripts/init_db.py
"""

import sqlite3
import sys
from pathlib import Path

# Make sure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ensure_dirs, load_config


def main() -> None:
    cfg = load_config()
    ensure_dirs()

    db_path = cfg["database"]["path"]
    schema  = Path("db/schema.sql").read_text()

    conn = sqlite3.connect(db_path)
    try:
        n_tables = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]

        if n_tables > 0:
            print(f"Database already initialised at {db_path} ({n_tables} tables). Nothing to do.")
        else:
            conn.executescript(schema)
            n_tables = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            print(f"Database initialised at {db_path} ({n_tables} tables).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
