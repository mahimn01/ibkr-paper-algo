"""US market rules — session hours, holiday calendar, MOC / LOC cutoffs.

Agents need to know three things before placing an order:
1. Are we in a regular-session window right now?
2. If not, how long until we are? (next open)
3. Are there special cutoffs (MOC, LOC, opening cross) we're about to miss?

This module gives all three from a pure-Python computation. No broker call,
no external data feed. Accuracy: good enough to guide intraday agents;
not a legal-authority calendar (for that, use exchange-published holiday
lists).

Times are all tz-aware in `America/New_York` — US equity / options /
futures market operate on ET.

Sessions modelled:
  - NYSE / NASDAQ / AMEX: 09:30–16:00 ET (regular), 04:00–09:30 pre,
    16:00–20:00 after. Half-days close 13:00 ET.
  - CBOE options: 09:30–16:00 ET (aligned w/ NYSE).
  - CME Globex (ES, NQ, CL, GC): 18:00 prior-day–17:00 ET, with a 60-min
    break 17:00–18:00 daily. Sunday opens 18:00.

Holidays modelled through the end of 2027 — update this list when Dec 31
2027 approaches. Half-days also enumerated (close 13:00 ET).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    UTC = timezone.utc
except Exception:  # pragma: no cover — fallback for pre-3.9 or stripped envs
    ET = timezone(timedelta(hours=-5))  # EST approximation, no DST
    UTC = timezone.utc


# ---------------------------------------------------------------------------
# Static calendar — NYSE / NASDAQ / AMEX holidays + half-days
# Sources: NYSE holiday schedule. Updated annually.
# ---------------------------------------------------------------------------

NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2025
    date(2025, 1, 1),   # New Year's
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
    # 2026
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),   # observed
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),
    date(2027, 5, 31),
    date(2027, 6, 18),  # observed (Juneteenth on Saturday)
    date(2027, 7, 5),   # observed
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24), # observed (Christmas on Saturday)
})

NYSE_HALF_DAYS: frozenset[date] = frozenset({
    date(2025, 7, 3),    # Day before Independence Day
    date(2025, 11, 28),  # Day after Thanksgiving
    date(2025, 12, 24),  # Christmas Eve
    date(2026, 11, 27),
    date(2026, 12, 24),
    date(2027, 11, 26),
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_et() -> datetime:
    return datetime.now(tz=UTC).astimezone(ET)


def _regular_open(d: date) -> datetime:
    return datetime.combine(d, time(9, 30), tzinfo=ET)


def _regular_close(d: date, half_day: bool = False) -> datetime:
    close = time(13, 0) if half_day else time(16, 0)
    return datetime.combine(d, close, tzinfo=ET)


def is_holiday(d: date) -> bool:
    return d in NYSE_HOLIDAYS


def is_half_day(d: date) -> bool:
    return d in NYSE_HALF_DAYS


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def is_trading_day(d: date) -> bool:
    return not is_weekend(d) and not is_holiday(d)


def is_us_equity_regular_open(when: datetime | None = None) -> bool:
    """True iff `when` (default: now in ET) is within the US equity
    regular session."""
    t = when.astimezone(ET) if when else now_et()
    d = t.date()
    if not is_trading_day(d):
        return False
    open_t = _regular_open(d)
    close_t = _regular_close(d, half_day=is_half_day(d))
    return open_t <= t < close_t


def next_trading_day(from_date: date) -> date:
    d = from_date
    while True:
        d = d + timedelta(days=1)
        if is_trading_day(d):
            return d


def next_open_et(when: datetime | None = None) -> datetime:
    """Return the next regular-session open boundary, ET."""
    t = when.astimezone(ET) if when else now_et()
    d = t.date()
    if is_trading_day(d):
        today_open = _regular_open(d)
        if t < today_open:
            return today_open
    return _regular_open(next_trading_day(d))


def next_close_et(when: datetime | None = None) -> datetime:
    """Return the next regular-session close boundary, ET.

    If we're already past today's close (or not a trading day), returns
    the *next trading day's* close.
    """
    t = when.astimezone(ET) if when else now_et()
    d = t.date()
    if is_trading_day(d):
        close_t = _regular_close(d, half_day=is_half_day(d))
        if t < close_t:
            return close_t
    nd = next_trading_day(d)
    return _regular_close(nd, half_day=is_half_day(nd))


# ---------------------------------------------------------------------------
# Session cutoffs — MOC / LOC / opening cross
# Reference: NYSE order-entry policy.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionCutoffs:
    """Order-entry cutoffs for the US equity session (ET).

    - moc: Market-on-close entry freeze (NYSE: 15:50 ET; IBKR routes
      reject MOC/LOC after). 10-min before close on half-days = 12:50.
    - loc: Limit-on-close entry freeze (NYSE: 15:50 ET, same as MOC).
    - opening_cross: MOO / LOO accepted until 09:28 ET (imbalance publish).
    - regular_open: 09:30 ET
    - regular_close: 16:00 ET (13:00 on half-days)
    """
    moc_cutoff: datetime
    loc_cutoff: datetime
    opening_cross_cutoff: datetime
    regular_open: datetime
    regular_close: datetime


def session_cutoffs(d: date) -> SessionCutoffs | None:
    """Compute today's session cutoffs. Returns None on non-trading days."""
    if not is_trading_day(d):
        return None
    half = is_half_day(d)
    close = _regular_close(d, half_day=half)
    return SessionCutoffs(
        moc_cutoff=close - timedelta(minutes=10),
        loc_cutoff=close - timedelta(minutes=10),
        opening_cross_cutoff=datetime.combine(d, time(9, 28), tzinfo=ET),
        regular_open=_regular_open(d),
        regular_close=close,
    )


# ---------------------------------------------------------------------------
# Summary — one dict for status / time command output
# ---------------------------------------------------------------------------

def market_state(when: datetime | None = None) -> dict:
    """One-shot summary used by `status` and `time` commands."""
    t = when.astimezone(ET) if when else now_et()
    d = t.date()
    state = {
        "et_now": t.isoformat(timespec="seconds"),
        "utc_now": t.astimezone(UTC).isoformat(timespec="seconds"),
        "et_date": d.isoformat(),
        "weekday": t.strftime("%A"),
        "is_trading_day": is_trading_day(d),
        "is_holiday": is_holiday(d),
        "is_half_day": is_half_day(d),
        "us_equity_regular_session_open": is_us_equity_regular_open(t),
        "next_open_et": next_open_et(t).isoformat(timespec="seconds"),
        "next_close_et": next_close_et(t).isoformat(timespec="seconds"),
    }
    cutoffs = session_cutoffs(d)
    if cutoffs is not None:
        state["session_cutoffs_et"] = {
            "regular_open": cutoffs.regular_open.isoformat(timespec="seconds"),
            "regular_close": cutoffs.regular_close.isoformat(timespec="seconds"),
            "moc_cutoff": cutoffs.moc_cutoff.isoformat(timespec="seconds"),
            "loc_cutoff": cutoffs.loc_cutoff.isoformat(timespec="seconds"),
            "opening_cross_cutoff": cutoffs.opening_cross_cutoff.isoformat(timespec="seconds"),
        }
    else:
        state["session_cutoffs_et"] = None
    return state
