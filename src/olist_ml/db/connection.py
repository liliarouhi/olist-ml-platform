"""Thin wrapper around a psycopg connection with transaction handling."""
from __future__ import annotations

import psycopg

from olist_ml.config import DatabaseConfig


class Database:


    def __init__(self, config: DatabaseConfig | None = None, autocommit: bool = False) -> None:
        self._config = config or DatabaseConfig()
        self._autocommit = autocommit
        self._conn: psycopg.Connection | None = None

    def __enter__(self) -> Database:
        self._conn = psycopg.connect(self._config.conninfo(), autocommit=self._autocommit)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is None:
            return
        if not self._autocommit:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        self._conn.close()
        self._conn = None

    @property
    def connection(self) -> psycopg.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not open; use 'with Database() as db:'")
        return self._conn

    def execute(self, sql, params: tuple | None = None) -> int:
        """Run one statement and return the number of affected rows."""
        with self.connection.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    def fetch_one(self, sql, params: tuple | None = None) -> tuple | None:
        """Run a query and return its first row."""
        with self.connection.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()