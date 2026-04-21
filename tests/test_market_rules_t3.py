"""Tests for T3.3 market_rules.py — US equity session calendar."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from trading_algo.market_rules import (
    ET,
    is_half_day,
    is_holiday,
    is_trading_day,
    is_us_equity_regular_open,
    is_weekend,
    market_state,
    next_close_et,
    next_open_et,
    session_cutoffs,
)


def _dt(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


class TestHolidayCalendar:
    def test_known_holidays(self) -> None:
        assert is_holiday(date(2025, 7, 4))
        assert is_holiday(date(2025, 12, 25))
        assert is_holiday(date(2026, 1, 1))
        assert not is_holiday(date(2025, 7, 3))

    def test_half_days(self) -> None:
        assert is_half_day(date(2025, 11, 28))  # Day after Thanksgiving
        assert not is_half_day(date(2025, 11, 27))  # Thanksgiving itself (full holiday)

    def test_is_trading_day(self) -> None:
        assert is_trading_day(date(2025, 7, 7))   # Monday
        assert not is_trading_day(date(2025, 7, 5))   # Saturday
        assert not is_trading_day(date(2025, 7, 6))   # Sunday
        assert not is_trading_day(date(2025, 7, 4))   # Independence


class TestRegularSessionWindow:
    def test_midday_is_open(self) -> None:
        assert is_us_equity_regular_open(_dt(2025, 7, 7, 12, 0))

    def test_premarket_is_closed(self) -> None:
        assert not is_us_equity_regular_open(_dt(2025, 7, 7, 8, 0))

    def test_afterhours_is_closed(self) -> None:
        assert not is_us_equity_regular_open(_dt(2025, 7, 7, 17, 0))

    def test_open_boundary(self) -> None:
        assert is_us_equity_regular_open(_dt(2025, 7, 7, 9, 30))
        assert not is_us_equity_regular_open(_dt(2025, 7, 7, 9, 29))

    def test_close_boundary(self) -> None:
        assert is_us_equity_regular_open(_dt(2025, 7, 7, 15, 59))
        assert not is_us_equity_regular_open(_dt(2025, 7, 7, 16, 0))

    def test_weekend_closed(self) -> None:
        assert not is_us_equity_regular_open(_dt(2025, 7, 5, 12, 0))

    def test_holiday_closed(self) -> None:
        assert not is_us_equity_regular_open(_dt(2025, 7, 4, 12, 0))

    def test_half_day_closes_at_13(self) -> None:
        # 2025-11-28 is day-after-Thanksgiving half day
        assert is_us_equity_regular_open(_dt(2025, 11, 28, 12, 0))
        assert not is_us_equity_regular_open(_dt(2025, 11, 28, 13, 0))


class TestNextOpenClose:
    def test_next_open_before_open(self) -> None:
        t = _dt(2025, 7, 7, 8, 0)
        assert next_open_et(t) == _dt(2025, 7, 7, 9, 30)

    def test_next_open_after_close(self) -> None:
        t = _dt(2025, 7, 7, 17, 0)  # Mon after close
        assert next_open_et(t) == _dt(2025, 7, 8, 9, 30)

    def test_next_open_skips_holiday(self) -> None:
        t = _dt(2025, 7, 3, 17, 0)  # Thu after close, Fri is holiday
        assert next_open_et(t) == _dt(2025, 7, 7, 9, 30)  # Mon

    def test_next_close_during_session(self) -> None:
        t = _dt(2025, 7, 7, 12, 0)
        assert next_close_et(t) == _dt(2025, 7, 7, 16, 0)

    def test_next_close_on_half_day(self) -> None:
        t = _dt(2025, 11, 28, 10, 0)
        assert next_close_et(t) == _dt(2025, 11, 28, 13, 0)


class TestSessionCutoffs:
    def test_regular_day_cutoffs(self) -> None:
        c = session_cutoffs(date(2025, 7, 7))
        assert c is not None
        assert c.regular_open.hour == 9 and c.regular_open.minute == 30
        assert c.regular_close.hour == 16
        assert c.moc_cutoff.hour == 15 and c.moc_cutoff.minute == 50
        assert c.loc_cutoff == c.moc_cutoff
        assert c.opening_cross_cutoff.hour == 9 and c.opening_cross_cutoff.minute == 28

    def test_half_day_cutoffs(self) -> None:
        c = session_cutoffs(date(2025, 11, 28))
        assert c is not None
        assert c.regular_close.hour == 13
        assert c.moc_cutoff.hour == 12 and c.moc_cutoff.minute == 50

    def test_holiday_returns_none(self) -> None:
        assert session_cutoffs(date(2025, 7, 4)) is None


class TestMarketState:
    def test_shape(self) -> None:
        s = market_state(_dt(2025, 7, 7, 12, 0))
        for k in ("et_now", "utc_now", "et_date", "weekday",
                  "is_trading_day", "is_holiday", "is_half_day",
                  "us_equity_regular_session_open",
                  "next_open_et", "next_close_et", "session_cutoffs_et"):
            assert k in s

    def test_holiday_has_no_cutoffs(self) -> None:
        s = market_state(_dt(2025, 7, 4, 12, 0))
        assert s["session_cutoffs_et"] is None
        assert s["us_equity_regular_session_open"] is False
