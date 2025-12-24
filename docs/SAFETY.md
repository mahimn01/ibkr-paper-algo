# Safety Model

This project is designed so you have to *try hard* to accidentally send paper orders, and even harder to ever hit live.

## Paper-only guard (always on)

`IBKRBroker` refuses to continue unless the connected session looks like **Paper Trading** by checking `managedAccounts()` and requiring all accounts start with `DU`.

This is enforced in code and is intentionally not configurable via environment variables.

## Order send gates (must all pass)

To send any IBKR order:

1. `TRADING_DRY_RUN=false`
2. `TRADING_LIVE_ENABLED=true`
3. `TRADING_ORDER_TOKEN` is set
4. `--confirm-token` matches `TRADING_ORDER_TOKEN`

If any gate fails, the order is blocked (dry-run logs/stages but does not send).

## Recommended workflow

1. Develop with `TRADING_DRY_RUN=true` and `TRADING_DB_PATH=...`
2. Run paper smoke tests and validate reconciliation/tracking
3. Only enable send-gates for short, intentional sessions

