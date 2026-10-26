# 04 — Transactions and Isolation

Script: `scripts/isolation_experiments.py`. Two concurrent sessions (T1, T2) reproduce classic
concurrency anomalies on a temporary schema. Raw results: `docs/figures/isolation_results.csv`.

## Background
A transaction is a unit of work with the ACID properties (Atomicity, Consistency, Isolation,
Durability). Isolation defines what concurrent transactions can see of each other.
PostgreSQL implements isolation with MVCC (Multi-Version Concurrency Control): each row can have
several versions, and every transaction reads the versions that match its snapshot, so readers
never block writers.

## Results

| Anomaly | Read Committed | Repeatable Read | Serializable |
|---|---|---|---|
| Dirty read | prevented (not possible in PostgreSQL) | prevented | prevented |
| Non-repeatable read | **occurs**: T1 read 100, then 50 | prevented: 100, then 100 | prevented |
| Phantom read | **occurs**: 2 rows, then 3 | prevented: 2, then 2 | prevented |
| Lost update (read, then write) | **occurs**: two +10 deposits on 100 gave 110 | detected: T2 aborted, must retry | detected |
| Write skew (on-call doctors) | occurs | **occurs**: 0 doctors left on call | prevented: T2 aborted, 1 on call |

Deadlock (Read Committed): T1 locked row 1 and waited for row 2, T2 locked row 2 and waited for
row 1. PostgreSQL detected the cycle after `deadlock_timeout` (1 s) and aborted T1, the session
that had been waiting longest, so T2 could finish.

## Findings
- **Read Committed** (the default) sees every committed change at statement level, so values and
  row counts can change inside one transaction.
- **Repeatable Read** works on one snapshot for the whole transaction. In PostgreSQL it also
  prevents phantom reads, which the SQL standard would allow at this level. Conflicting updates
  to the same row are resolved by aborting the second writer (first-updater-wins).
- **Write skew survives Repeatable Read:** each transaction changed a *different* row, so there
  was no write conflict, yet together they broke the rule "at least one doctor on call".
  Only **Serializable** (Serializable Snapshot Isolation) detects the read/write dependency cycle.
- **Serialization failures are expected, not bugs:** applications using Repeatable Read or
  Serializable must retry the aborted transaction.

## Practical rules
- In Read Committed, avoid read-then-write logic in the application. Use an atomic update
  (`UPDATE accounts SET balance = balance + 10`) or lock the row first (`SELECT ... FOR UPDATE`).
- Acquire locks in a consistent order (for example by primary key) to prevent deadlocks.
- Use Serializable when correctness depends on rules spanning several rows, together with retries.

## In this project
`CoreLoader` (`src/olist_ml/etl/loader.py`) runs the whole staging → core load in a single
transaction, so a failure rolls everything back (atomicity). It is the only writer, so the
default Read Committed level is sufficient.