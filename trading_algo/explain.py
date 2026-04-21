"""Per-command structured descriptions (`--explain`).

`--explain` emits a structured description of what a command would do,
without making any IBKR call or side effect. This differs from IBKR's
`whatIfOrder` (which DOES make an RPC round-trip for margin preview) —
`--explain` is pure-local, so the agent can enumerate preconditions and
side effects offline.

Shape (per command):

    {
      "action": "place_order",
      "side_effects": ["live order transmitted to IBKR OMS", ...],
      "preconditions": ["TWS/Gateway connected", "market open", ...],
      "reversibility": "cancellable while Submitted; post-fill requires offset",
      "idempotency": "orderRef-based orderbook pre-check on --idempotency-key",
      "rate_limit_bucket": "orders (~50 msg/s aggregate)",
      "requires_market_data_subscription": true | false,
      "notes": [...]
    }
"""

from __future__ import annotations

from typing import Any


_EXPLANATIONS: dict[str, dict[str, Any]] = {
    # ----------------------------------------------------------------
    # Connection / meta
    # ----------------------------------------------------------------
    "connect": {
        "action": "open an ib_async connection to TWS / IB Gateway",
        "side_effects": [
            "opens a TCP socket to host:port (default 127.0.0.1:7497 paper)",
            "consumes one client_id slot",
        ],
        "preconditions": [
            "TWS or IB Gateway running and logged in",
            "API → Settings → Enable ActiveX and Socket Clients checked",
            "client_id not already in use",
        ],
        "reversibility": "disconnect via IB.disconnect()",
        "idempotency": "reconnecting with the same client_id kicks the prior session",
        "rate_limit_bucket": "n/a",
    },
    "time": {
        "action": "GET server time (local ib.reqCurrentTime())",
        "side_effects": ["none"],
        "preconditions": ["connected"],
        "reversibility": "n/a",
        "idempotency": "safe",
        "rate_limit_bucket": "general (~50 msg/s)",
    },

    # ----------------------------------------------------------------
    # Account / portfolio — read
    # ----------------------------------------------------------------
    "accounts": {
        "action": "list managed account ids (ib.managedAccounts())",
        "side_effects": ["none"],
        "preconditions": ["connected"],
        "reversibility": "n/a",
        "idempotency": "safe",
    },
    "positions": {
        "action": "list current positions (ib.positions())",
        "side_effects": ["none"],
        "preconditions": ["connected"],
        "reversibility": "n/a",
        "idempotency": "safe",
        "notes": ["--summary rolls up long/short + per-account + largest"],
    },
    "portfolio": {
        "action": "list portfolio items with MTM (ib.portfolio())",
        "side_effects": ["subscribes to account updates (one-shot via reqAccountUpdates)"],
        "preconditions": ["connected"],
        "reversibility": "n/a",
        "idempotency": "safe",
        "notes": ["--summary returns best/worst performer + total PnL"],
    },
    "pnl": {
        "action": "subscribe to account PnL (ib.reqPnL())",
        "side_effects": ["opens PnL subscription — release via cancelPnL"],
        "preconditions": ["connected"],
        "reversibility": "cancelPnL",
        "idempotency": "safe",
    },
    "summary": {
        "action": "account summary tags (ib.accountSummary())",
        "side_effects": ["none"],
        "preconditions": ["connected"],
        "reversibility": "n/a",
        "idempotency": "safe",
    },

    # ----------------------------------------------------------------
    # Market data
    # ----------------------------------------------------------------
    "quote": {
        "action": "snapshot quote (reqTickers for STK; includes greeks for OPT)",
        "side_effects": ["none (one-shot; no streaming subscription retained)"],
        "preconditions": [
            "connected",
            "market-data subscription for the instrument's primary exchange",
        ],
        "reversibility": "n/a",
        "idempotency": "safe",
        "rate_limit_bucket": "market-data lines (100 concurrent on free tier)",
        "requires_market_data_subscription": True,
        "notes": [
            "If you see error 354 (subscription missing), add the subscription",
            "in Client Portal → Settings → Market Data Subscriptions.",
        ],
    },
    "quotes": {
        "action": "batch snapshot for multiple STK symbols via reqTickers",
        "side_effects": ["none"],
        "preconditions": ["connected", "market-data subscriptions"],
        "reversibility": "n/a",
        "idempotency": "safe",
        "requires_market_data_subscription": True,
    },
    "stream": {
        "action": "open tick-by-tick or bar stream (reqTickByTickData / reqRealTimeBars)",
        "side_effects": [
            "holds a market-data line subscription",
            "emits NDJSON to stdout until --duration expires",
        ],
        "preconditions": ["connected", "market-data subscription"],
        "reversibility": "Ctrl+C to close",
        "idempotency": "re-running subscribes again; not a problem but costs a line",
        "requires_market_data_subscription": True,
        "notes": [
            "IBKR hard caps 100 concurrent market-data lines on free tier.",
            "For programmatic tick capture, prefer --buffer-to if supported.",
        ],
    },
    "history": {
        "action": "historical bars (reqHistoricalData)",
        "side_effects": ["subject to IBKR pacing limits"],
        "preconditions": ["connected", "historical data subscription"],
        "reversibility": "n/a",
        "idempotency": "safe",
        "rate_limit_bucket": "historical (6/2s + 60 identical/10min pacing)",
        "requires_market_data_subscription": True,
        "notes": [
            "Per-bar-size duration caps: 1s→33min, 1m→1day, 1h→1month, 1d→1year.",
            "Exceeding triggers errorCode 162/165 — back off 10-60s, then retry.",
        ],
    },

    # ----------------------------------------------------------------
    # Orders — write
    # ----------------------------------------------------------------
    "place-order": {
        "action": "placeOrder on TWS/Gateway",
        "side_effects": [
            "live order transmitted to IBKR OMS",
            "routes to exchange per smart-routing rules",
            "capital reservation per account's margin model",
            "position modified on fill",
        ],
        "preconditions": [
            "connected (paper or live per --paper / --live / env)",
            "contract qualifies (errorCode 200 if no security definition)",
            "sufficient buying power",
            "account enabled for the instrument type / exchange",
            "`TRADING_ALLOW_LIVE=true` AND `TRADING_LIVE_ENABLED=true` for live accounts",
        ],
        "reversibility": (
            "cancellable while Submitted / PreSubmitted; post-fill requires "
            "an offset trade"
        ),
        "idempotency": (
            "orderRef-based cross-process dedup via --idempotency-key. "
            "Derives a deterministic orderRef from the key; "
            "IdempotentOrderPlacer queries ib.openTrades() + "
            "ib.reqCompletedOrders() before transmitting."
        ),
        "rate_limit_bucket": "orders (~50 msg/s aggregate)",
        "notes": [
            "--dry-run stages without transmitting (our safety rail, NOT IBKR's what-if).",
            "For margin preview, use `whatif` (IBKR's whatIfOrder RPC).",
            "orderRef is free-text (≤40 chars), survives TWS restart, reported in Flex.",
        ],
    },
    "place-bracket": {
        "action": "parent LMT entry + take-profit LMT + stop-loss STP as OCA bracket",
        "side_effects": [
            "three orders transmitted atomically (bracket semantics)",
            "OCA group enforces one-cancels-all on the protective legs",
        ],
        "preconditions": [
            "connected + sufficient buying power",
            "instrument allows bracket order type on its exchange",
        ],
        "reversibility": "cancel the parent → children auto-cancel via OCA",
        "idempotency": "supply --idempotency-key; we derive one orderRef per leg",
        "rate_limit_bucket": "orders (three messages)",
    },
    "cancel-order": {
        "action": "cancelOrder by orderId",
        "side_effects": ["the order moves to ApiCancelled / Cancelled"],
        "preconditions": ["order exists and is Submitted / PreSubmitted"],
        "reversibility": "place a fresh order to replace",
        "idempotency": "cancelling a terminal order is a no-op + warning",
        "rate_limit_bucket": "orders",
    },
    "cancel-all": {
        "action": "cancel every Submitted / PreSubmitted order (reqGlobalCancel)",
        "side_effects": [
            "DESTRUCTIVE: wipes the open book atomically",
            "OCA groups unwind",
        ],
        "preconditions": ["connected"],
        "reversibility": "n/a",
        "idempotency": "safe — a second call cancels nothing more",
        "rate_limit_bucket": "orders",
        "notes": [
            "Wave T3 adds --confirm-panic to require a distinct token beyond --yes.",
        ],
    },
    "modify-order": {
        "action": "placeOrder with existing orderId and updated fields",
        "side_effects": ["order re-transmitted to exchange"],
        "preconditions": ["order active, not terminal"],
        "reversibility": "cancel to revert; modify again with prior values",
        "idempotency": (
            "Wave T3 will add a per-order modification counter. Today, "
            "running modify many times on the same order risks exchange-"
            "side throttling."
        ),
        "rate_limit_bucket": "orders",
    },
    "whatif": {
        "action": "margin preview via whatIfOrder RPC (no transmission)",
        "side_effects": ["one RPC round-trip; no order, no capital reservation"],
        "preconditions": ["connected"],
        "reversibility": "n/a — no side effect",
        "idempotency": "safe",
        "notes": [
            "This IS an IBKR API call (costs a message) — differs from our "
            "local --explain meta-flag which is purely descriptive.",
        ],
    },

    # ----------------------------------------------------------------
    # Engine / autorun
    # ----------------------------------------------------------------
    "run": {
        "action": "run the strategy polling loop (Engine.run_forever)",
        "side_effects": [
            "places orders per strategy output",
            "persists to TRADING_DB_PATH SqliteStore",
            "audits to data/audit/*.jsonl",
        ],
        "preconditions": [
            "connected",
            "TRADING_LIVE_ENABLED=true for live accounts",
            "HALTED sentinel not set",
        ],
        "reversibility": "Ctrl+C / halt sentinel stops orders",
        "idempotency": "n/a — long-running loop",
    },

    # ----------------------------------------------------------------
    # Admin
    # ----------------------------------------------------------------
    "halt": {
        "action": "write data/HALTED sentinel",
        "side_effects": [
            "every subsequent write command refuses with exit 11 (HALTED)",
        ],
        "preconditions": ["none"],
        "reversibility": "`resume --confirm-resume` clears it",
        "idempotency": "repeated halt overwrites with new reason/expiry",
    },
    "resume": {
        "action": "remove the data/HALTED sentinel",
        "side_effects": ["writes resume"],
        "preconditions": ["--confirm-resume explicitly set"],
        "reversibility": "halt again to re-block writes",
        "idempotency": "safe — removing an absent sentinel is a no-op",
    },

    # ----------------------------------------------------------------
    # OMS
    # ----------------------------------------------------------------
    "oms-reconcile": {
        "action": "poll all non-terminal orders in SqliteStore and fetch live IBKR status",
        "side_effects": ["logs status-change events to SqliteStore"],
        "preconditions": ["TRADING_DB_PATH set", "connected"],
        "reversibility": "n/a",
        "idempotency": "safe to run repeatedly",
    },
    "oms-track": {
        "action": "block until all tracked orders reach terminal state or timeout",
        "side_effects": ["persists status transitions; may take minutes"],
        "preconditions": ["TRADING_DB_PATH set", "connected"],
        "reversibility": "Ctrl+C to exit early",
        "idempotency": "safe",
    },

    # ----------------------------------------------------------------
    # Flex (flex_tool.py)
    # ----------------------------------------------------------------
    "flex-send": {
        "action": "POST FlexService request to start generating a report",
        "side_effects": ["Flex server enqueues a report job"],
        "preconditions": ["FLEX_TOKEN env set", "FlexQueryId env set"],
        "reversibility": "n/a",
        "idempotency": "safe — Flex generates fresh reference ids",
    },
    "flex-poll": {
        "action": "poll Flex server for ready reports; cache XML to data/flex/",
        "side_effects": ["writes data/flex/*.xml on ready"],
        "preconditions": ["prior `flex-send` returned a reference id"],
        "reversibility": "n/a",
        "idempotency": "safe",
    },
    "flex-trades": {
        "action": "parse cached Flex activity XML → trades rows",
        "side_effects": ["none"],
        "preconditions": ["cached XML in data/flex/"],
        "reversibility": "n/a",
        "idempotency": "safe",
    },
}


_FALLBACK = {
    "action": "(command description not yet filled in)",
    "side_effects": ["see README / CLI docs"],
    "preconditions": ["TWS/Gateway connected for IBKR commands"],
    "reversibility": "see README / CLI docs",
    "idempotency": "see README / CLI docs",
}


def explain(cmd: str) -> dict[str, Any]:
    """Return the explanation record for a subcommand. Never raises."""
    if cmd in _EXPLANATIONS:
        return dict(_EXPLANATIONS[cmd])
    return dict(_FALLBACK, command=cmd)


def all_explanations() -> dict[str, dict[str, Any]]:
    return {name: dict(body) for name, body in _EXPLANATIONS.items()}
