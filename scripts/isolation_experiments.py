"""Reproduce transaction isolation anomalies in PostgreSQL with two concurrent sessions.

Usage:  python scripts/isolation_experiments.py
Output: docs/figures/isolation_results.csv
All experiments run in a temporary schema `lab`, which is dropped at the end.
"""
from __future__ import annotations

import csv
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psycopg
from psycopg import errors

from olist_ml.config import DatabaseConfig

logger = logging.getLogger(__name__)
OUT_FILE = Path("docs/figures/isolation_results.csv")

# Fixed constants, never user input, so formatting them into SQL is safe.
READ_COMMITTED = "READ COMMITTED"
REPEATABLE_READ = "REPEATABLE READ"
SERIALIZABLE = "SERIALIZABLE"


@dataclass(frozen=True)
class ExperimentResult:
    experiment: str
    isolation_level: str
    problem_observed: bool
    details: str


class Session:
    """One database session with explicit BEGIN / COMMIT, so every step is visible in the code."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._conn = psycopg.connect(DatabaseConfig().conninfo(), autocommit=True)

    def begin(self, level: str) -> None:
        self._conn.execute(f"BEGIN ISOLATION LEVEL {level}")

    def scalar(self, sql: str, params: tuple | None = None):
        return self._conn.execute(sql, params).fetchone()[0]

    def run(self, sql: str, params: tuple | None = None) -> None:
        self._conn.execute(sql, params)

    def commit(self) -> None:
        self._conn.execute("COMMIT")

    def rollback(self) -> None:
        self._conn.execute("ROLLBACK")

    def close(self) -> None:
        self._conn.close()


class IsolationLab:
    """Runs each anomaly scenario with two sessions, T1 and T2, plus an admin session for setup."""

    def __init__(self) -> None:
        self.admin = Session("admin")
        self.t1 = Session("T1")
        self.t2 = Session("T2")

    def _reset(self) -> None:
        self.admin.run("CREATE SCHEMA IF NOT EXISTS lab")
        self.admin.run("DROP TABLE IF EXISTS lab.accounts, lab.doctors")
        self.admin.run("CREATE TABLE lab.accounts (id int PRIMARY KEY, balance int NOT NULL)")
        self.admin.run("INSERT INTO lab.accounts VALUES (1, 100), (2, 100)")
        self.admin.run("CREATE TABLE lab.doctors (name text PRIMARY KEY, on_call boolean NOT NULL)")
        self.admin.run("INSERT INTO lab.doctors VALUES ('alice', true), ('bob', true)")

    def non_repeatable_read(self, level: str) -> ExperimentResult:
        self._reset()
        self.t1.begin(level)
        first = self.t1.scalar("SELECT balance FROM lab.accounts WHERE id = 1")
        self.t2.begin(level)
        self.t2.run("UPDATE lab.accounts SET balance = 50 WHERE id = 1")
        self.t2.commit()
        second = self.t1.scalar("SELECT balance FROM lab.accounts WHERE id = 1")
        self.t1.commit()
        return ExperimentResult("non-repeatable read", level, first != second,
                                f"T1 read balance {first}, then {second}")

    def phantom_read(self, level: str) -> ExperimentResult:
        self._reset()
        query = "SELECT count(*) FROM lab.accounts WHERE balance >= 100"
        self.t1.begin(level)
        first = self.t1.scalar(query)
        self.t2.begin(level)
        self.t2.run("INSERT INTO lab.accounts VALUES (3, 200)")
        self.t2.commit()
        second = self.t1.scalar(query)
        self.t1.commit()
        return ExperimentResult("phantom read", level, first != second,
                                f"T1 counted {first} rows, then {second}")

    def lost_update(self, level: str) -> ExperimentResult:
        self._reset()
        query = "SELECT balance FROM lab.accounts WHERE id = 1"
        self.t1.begin(level)
        self.t2.begin(level)
        seen_by_t1 = self.t1.scalar(query)
        seen_by_t2 = self.t2.scalar(query)
        self.t1.run("UPDATE lab.accounts SET balance = %s WHERE id = 1", (seen_by_t1 + 10,))
        self.t1.commit()
        try:
            self.t2.run("UPDATE lab.accounts SET balance = %s WHERE id = 1", (seen_by_t2 + 10,))
            self.t2.commit()
            outcome = "both committed"
        except errors.SerializationFailure:
            self.t2.rollback()
            outcome = "T2 aborted with a serialization failure (must retry)"
        final = self.admin.scalar(query)
        lost = outcome == "both committed" and final != 120
        return ExperimentResult("lost update", level, lost,
                                f"two +10 deposits on 100 gave {final}; {outcome}")

    def write_skew(self, level: str) -> ExperimentResult:
        self._reset()
        query = "SELECT count(*) FROM lab.doctors WHERE on_call"
        self.t1.begin(level)
        self.t2.begin(level)
        seen = {self.t1: self.t1.scalar(query), self.t2: self.t2.scalar(query)}
        outcomes = []
        for session, doctor in ((self.t1, "alice"), (self.t2, "bob")):
            try:
                if seen[session] >= 2:  # rule: someone else is still on call, so I may leave
                    session.run("UPDATE lab.doctors SET on_call = false WHERE name = %s", (doctor,))
                session.commit()
                outcomes.append(f"{session.name} committed")
            except errors.SerializationFailure:
                session.rollback()
                outcomes.append(f"{session.name} aborted")
        remaining = self.admin.scalar(query)
        return ExperimentResult("write skew", level, remaining == 0,
                                f"doctors on call afterwards: {remaining}; {', '.join(outcomes)}")

    def deadlock(self) -> ExperimentResult:
        self._reset()
        self.t1.begin(READ_COMMITTED)
        self.t2.begin(READ_COMMITTED)
        self.t1.run("UPDATE lab.accounts SET balance = balance - 10 WHERE id = 1")  # T1 locks row 1
        self.t2.run("UPDATE lab.accounts SET balance = balance - 10 WHERE id = 2")  # T2 locks row 2
        victims: list[str] = []

        def t1_wants_row_2() -> None:
            try:
                self.t1.run("UPDATE lab.accounts SET balance = balance + 10 WHERE id = 2")
            except errors.DeadlockDetected:
                victims.append("T1")
                self.t1.rollback()

        worker = threading.Thread(target=t1_wants_row_2)
        worker.start()
        time.sleep(0.5)  # T1 is now waiting for T2's lock
        try:
            self.t2.run("UPDATE lab.accounts SET balance = balance + 10 WHERE id = 1")  # cycle
        except errors.DeadlockDetected:
            victims.append("T2")
            self.t2.rollback()
        worker.join(timeout=10)
        self.t1.rollback()
        self.t2.rollback()
        return ExperimentResult("deadlock", READ_COMMITTED, bool(victims),
                                f"PostgreSQL detected the lock cycle and aborted {', '.join(victims)}")

    def close(self) -> None:
        self.admin.run("DROP SCHEMA IF EXISTS lab CASCADE")
        for session in (self.t1, self.t2, self.admin):
            session.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lab = IsolationLab()
    results: list[ExperimentResult] = []
    try:
        for level in (READ_COMMITTED, REPEATABLE_READ):
            results.append(lab.non_repeatable_read(level))
            results.append(lab.phantom_read(level))
            results.append(lab.lost_update(level))
        for level in (REPEATABLE_READ, SERIALIZABLE):
            results.append(lab.write_skew(level))
        results.append(lab.deadlock())
    finally:
        lab.close()

    for r in results:
        logger.info("%-20s %-16s problem=%-5s %s",
                    r.experiment, r.isolation_level, r.problem_observed, r.details)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


if __name__ == "__main__":
    main()