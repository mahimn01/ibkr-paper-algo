# SQLite Audit Trail

Enable by setting `TRADING_DB_PATH=trading_audit.sqlite3`.

Tables (see `trading_algo/persistence.py`):

- `runs`: start/end timestamps and config snapshot
- `decisions`: per-strategy decision log (accepted/rejected intents)
- `orders`: submitted orders (plus denormalized fields for quick queries)
- `order_status_events`: status snapshots over time (Submitted/Filled/Cancelled/etc.)
- `errors`: errors raised while running/processing

This DB is used by OMS reconciliation:
- On startup, non-terminal order ids are read from `orders.status`
- They are reconciled with broker open orders and `order_status_events` are appended

