"""Row projection (--fields) + summary rollups (--summary) for IBKR data.

Agent-driven calls frequently care about a small subset of a command's
response:

- `summary` for today's orders → `{total, by_status, open_count,
  oldest_open_timestamp, total_buy_value, total_sell_value}`.
- `summary` for positions → `{count, total_net_liq, total_unrealized_pnl,
  total_realized_pnl, worst, best}`.
- `summary` for portfolio → `{count, invested, value, day_pnl, best, worst}`.

Without `--summary` the list endpoints emit full rows (15+ fields each),
which is 10-50 KB of JSON that the agent then has to hold in context at
cost. `--summary` replaces with a ~500-byte rollup.

`--fields a,b,c` keeps only the named columns on list-returning commands.
Used when the agent knows exactly which fields it needs (e.g.
`orders --fields orderId,status,filledQuantity,avgFillPrice`).
"""

from __future__ import annotations

from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Field projection
# ---------------------------------------------------------------------------

def project_rows(rows: list[dict], fields: list[str] | None) -> list[dict]:
    """Return a new list where each row keeps only the named keys.

    Missing fields are included as `None` so the output shape stays stable
    across rows (important for CSV header consistency).  Pass `None` or an
    empty list to get the rows unchanged.
    """
    if not fields:
        return rows
    return [{f: r.get(f) for f in fields} for r in rows]


def parse_fields(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    out = [s.strip() for s in raw.split(",") if s.strip()]
    return out or None


# ---------------------------------------------------------------------------
# Summary rollups
# ---------------------------------------------------------------------------

# IBKR terminal order statuses (per ib_async / IBKR docs).
_IBKR_TERMINAL_STATUSES = frozenset({
    "Filled", "Cancelled", "ApiCancelled", "Rejected", "Inactive",
})
_IBKR_ACTIVE_STATUSES = frozenset({
    "PreSubmitted", "Submitted", "PendingSubmit", "PendingCancel",
})


def summarize_orders(orders: list[dict]) -> dict:
    """Compact rollup of today's orders list.

    Works on both raw ib_async OpenOrder dicts and our own `OrderStatus`
    shape — extracts status / side / qty from whichever field is present.

    {
      total: int,
      by_status: {Submitted: N, Filled: N, ...},
      active_count: int,
      oldest_open_timestamp: str | null,
      total_buy_value: float,
      total_sell_value: float,
    }
    """
    if not orders:
        return {
            "total": 0, "by_status": {}, "active_count": 0,
            "oldest_open_timestamp": None,
            "total_buy_value": 0.0, "total_sell_value": 0.0,
        }

    def _status(o: dict) -> str:
        return (
            o.get("status")
            or (o.get("orderStatus") or {}).get("status")
            or "UNKNOWN"
        )

    def _side(o: dict) -> str:
        return (o.get("side") or (o.get("order") or {}).get("action") or "").upper()

    def _qty(o: dict) -> float:
        return float(
            o.get("totalQuantity")
            or o.get("quantity")
            or (o.get("order") or {}).get("totalQuantity")
            or 0
        )

    def _px(o: dict) -> float:
        return float(
            o.get("lmtPrice")
            or o.get("avgFillPrice")
            or (o.get("orderStatus") or {}).get("avgFillPrice")
            or (o.get("order") or {}).get("lmtPrice")
            or 0
        )

    def _ts(o: dict) -> str | None:
        for k in ("ts", "submissionTime", "orderTimestamp", "order_timestamp"):
            v = o.get(k)
            if v:
                return str(v)
        return None

    by_status = Counter(_status(o) for o in orders)
    active = [o for o in orders if _status(o) in _IBKR_ACTIVE_STATUSES]
    oldest = None
    if active:
        timestamps = [_ts(o) for o in active if _ts(o)]
        if timestamps:
            oldest = min(timestamps)

    buy_v = sum(_qty(o) * _px(o) for o in orders if _side(o) == "BUY")
    sell_v = sum(_qty(o) * _px(o) for o in orders if _side(o) == "SELL")

    return {
        "total": len(orders),
        "by_status": dict(by_status),
        "active_count": len(active),
        "oldest_open_timestamp": oldest,
        "total_buy_value": round(buy_v, 2),
        "total_sell_value": round(sell_v, 2),
    }


def summarize_positions(positions: list[dict]) -> dict:
    """Rollup for ib.positions() output.

    Expected shape (per position): {account, contract, position, avgCost}.
    Works also on our own Position dataclass (as_dict) with the same
    keys.

    {
      count: int,
      long_count: int,
      short_count: int,
      by_account: {accountId: count},
      total_abs_notional: float,   # |qty * avgCost| summed
      largest_position: {symbol, quantity, notional},
    }
    """
    if not positions:
        return {
            "count": 0, "long_count": 0, "short_count": 0,
            "by_account": {}, "total_abs_notional": 0.0,
            "largest_position": None,
        }

    def _qty(p: dict) -> float:
        return float(p.get("position") or p.get("quantity") or 0)

    def _avg_cost(p: dict) -> float:
        return float(p.get("avgCost") or p.get("avg_cost") or 0)

    def _symbol(p: dict) -> str:
        c = p.get("contract")
        if isinstance(c, dict):
            return str(c.get("symbol") or "?")
        if hasattr(c, "symbol"):
            return str(c.symbol)
        inst = p.get("instrument")
        if isinstance(inst, dict):
            return str(inst.get("symbol") or "?")
        return str(p.get("symbol") or "?")

    def _account(p: dict) -> str:
        return str(p.get("account") or p.get("accountId") or "default")

    long_n = sum(1 for p in positions if _qty(p) > 0)
    short_n = sum(1 for p in positions if _qty(p) < 0)
    by_account = Counter(_account(p) for p in positions)
    notionals = [(_symbol(p), abs(_qty(p) * _avg_cost(p))) for p in positions]
    total_notional = sum(n for _, n in notionals)
    largest = max(notionals, key=lambda x: x[1], default=None)
    largest_dict = None
    if largest is not None and largest[1] > 0:
        # Find the matching position for qty.
        for p in positions:
            if _symbol(p) == largest[0]:
                largest_dict = {
                    "symbol": largest[0],
                    "quantity": _qty(p),
                    "notional": round(largest[1], 2),
                }
                break

    return {
        "count": len(positions),
        "long_count": long_n,
        "short_count": short_n,
        "by_account": dict(by_account),
        "total_abs_notional": round(total_notional, 2),
        "largest_position": largest_dict,
    }


def summarize_portfolio(items: list[dict]) -> dict:
    """Rollup for ib.portfolio() output (PortfolioItem: contract, position,
    marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL).

    {
      count, total_market_value, total_unrealized_pnl, total_realized_pnl,
      best_performer: {symbol, pnl_pct}, worst_performer: {symbol, pnl_pct}
    }
    """
    if not items:
        return {
            "count": 0, "total_market_value": 0.0,
            "total_unrealized_pnl": 0.0, "total_realized_pnl": 0.0,
            "best_performer": None, "worst_performer": None,
        }

    def _sym(i: dict) -> str:
        c = i.get("contract")
        if isinstance(c, dict):
            return str(c.get("symbol") or "?")
        if hasattr(c, "symbol"):
            return str(c.symbol)
        return "?"

    def _mv(i: dict) -> float:
        return float(i.get("marketValue") or 0)

    def _urpnl(i: dict) -> float:
        return float(i.get("unrealizedPNL") or 0)

    def _rpnl(i: dict) -> float:
        return float(i.get("realizedPNL") or 0)

    def _avg_cost(i: dict) -> float:
        return float(i.get("averageCost") or 0)

    def _mark(i: dict) -> float:
        return float(i.get("marketPrice") or 0)

    best = (None, float("-inf"))
    worst = (None, float("inf"))
    for i in items:
        avg = _avg_cost(i)
        mp = _mark(i)
        if avg > 0:
            pnl_pct = (mp - avg) / avg * 100
        else:
            pnl_pct = 0.0
        sym = _sym(i)
        if pnl_pct > best[1]:
            best = (sym, pnl_pct)
        if pnl_pct < worst[1]:
            worst = (sym, pnl_pct)

    return {
        "count": len(items),
        "total_market_value": round(sum(_mv(i) for i in items), 2),
        "total_unrealized_pnl": round(sum(_urpnl(i) for i in items), 2),
        "total_realized_pnl": round(sum(_rpnl(i) for i in items), 2),
        "best_performer": (
            {"symbol": best[0], "pnl_pct": round(best[1], 2)}
            if best[0] else None
        ),
        "worst_performer": (
            {"symbol": worst[0], "pnl_pct": round(worst[1], 2)}
            if worst[0] else None
        ),
    }


def summarize_executions(trades: list[dict]) -> dict:
    """Rollup for today's executions (ib.fills() / ib.trades())."""
    if not trades:
        return {
            "count": 0, "by_side": {},
            "total_commission": 0.0, "total_value": 0.0,
        }

    def _side(t: dict) -> str:
        exec_ = t.get("execution") or {}
        return str(exec_.get("side") or t.get("side") or "?").upper()

    def _value(t: dict) -> float:
        exec_ = t.get("execution") or {}
        px = float(exec_.get("avgPrice") or exec_.get("price") or 0)
        qty = float(exec_.get("shares") or exec_.get("quantity") or 0)
        return px * qty

    def _commission(t: dict) -> float:
        cr = t.get("commissionReport") or {}
        return float(cr.get("commission") or 0)

    return {
        "count": len(trades),
        "by_side": dict(Counter(_side(t) for t in trades)),
        "total_commission": round(sum(_commission(t) for t in trades), 2),
        "total_value": round(sum(_value(t) for t in trades), 2),
    }
