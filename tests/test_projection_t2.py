"""Tests for projection helpers (trading-algo T2)."""

from __future__ import annotations

import pytest

from trading_algo.projection import (
    parse_fields,
    project_rows,
    summarize_executions,
    summarize_orders,
    summarize_portfolio,
    summarize_positions,
)


class TestParseFields:
    def test_basic(self) -> None:
        assert parse_fields("a,b,c") == ["a", "b", "c"]

    def test_whitespace(self) -> None:
        assert parse_fields(" a , b ,c ") == ["a", "b", "c"]

    def test_empty(self) -> None:
        assert parse_fields("") is None
        assert parse_fields(None) is None


class TestProjectRows:
    def test_keeps_named(self) -> None:
        rows = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
        assert project_rows(rows, ["a", "c"]) == [{"a": 1, "c": 3}, {"a": 4, "c": 6}]

    def test_missing_is_none(self) -> None:
        out = project_rows([{"a": 1}, {"a": 2, "b": 9}], ["a", "b"])
        assert out == [{"a": 1, "b": None}, {"a": 2, "b": 9}]

    def test_passthrough(self) -> None:
        rows = [{"a": 1}]
        assert project_rows(rows, None) is rows


class TestSummarizeOrders:
    def test_empty(self) -> None:
        s = summarize_orders([])
        assert s["total"] == 0

    def test_mixed_ibkr_statuses(self) -> None:
        orders = [
            {"status": "Submitted", "side": "BUY", "quantity": 10,
             "lmtPrice": 150, "ts": "2026-04-21T10:00:00"},
            {"status": "Filled", "side": "BUY", "quantity": 5,
             "avgFillPrice": 155},
            {"status": "Cancelled", "side": "SELL", "quantity": 1,
             "lmtPrice": 160},
            {"status": "PreSubmitted", "side": "SELL", "quantity": 3,
             "lmtPrice": 158, "ts": "2026-04-21T10:05:00"},
        ]
        s = summarize_orders(orders)
        assert s["total"] == 4
        assert s["by_status"] == {"Submitted": 1, "Filled": 1,
                                   "Cancelled": 1, "PreSubmitted": 1}
        assert s["active_count"] == 2  # Submitted + PreSubmitted
        assert s["oldest_open_timestamp"] == "2026-04-21T10:00:00"
        assert s["total_buy_value"] == 10 * 150 + 5 * 155
        assert s["total_sell_value"] == 1 * 160 + 3 * 158

    def test_reads_ib_async_shape(self) -> None:
        """Real ib_async Trade objects have nested order + orderStatus."""
        orders = [
            {
                "order": {"action": "BUY", "totalQuantity": 10, "lmtPrice": 100},
                "orderStatus": {"status": "Submitted"},
            },
            {
                "order": {"action": "SELL", "totalQuantity": 5, "lmtPrice": 105},
                "orderStatus": {"status": "Filled", "avgFillPrice": 106},
            },
        ]
        s = summarize_orders(orders)
        assert s["by_status"] == {"Submitted": 1, "Filled": 1}


class TestSummarizePositions:
    def test_empty(self) -> None:
        s = summarize_positions([])
        assert s["count"] == 0

    def test_long_short_mix(self) -> None:
        positions = [
            {"account": "DU1", "contract": {"symbol": "AAPL"},
             "position": 100, "avgCost": 150},
            {"account": "DU1", "contract": {"symbol": "MSFT"},
             "position": -50, "avgCost": 400},
            {"account": "DU2", "contract": {"symbol": "NVDA"},
             "position": 10, "avgCost": 1000},
        ]
        s = summarize_positions(positions)
        assert s["count"] == 3
        assert s["long_count"] == 2
        assert s["short_count"] == 1
        assert s["by_account"] == {"DU1": 2, "DU2": 1}
        # Biggest notional: MSFT at 50 * 400 = 20000.
        assert s["largest_position"]["symbol"] == "MSFT"
        assert s["largest_position"]["notional"] == 20000

    def test_contract_as_object(self) -> None:
        """ib_async returns contract as a dataclass-like object."""
        class FakeContract:
            symbol = "AAPL"
        positions = [{"contract": FakeContract(), "position": 10, "avgCost": 150}]
        s = summarize_positions(positions)
        assert s["count"] == 1
        assert s["largest_position"]["symbol"] == "AAPL"


class TestSummarizePortfolio:
    def test_empty(self) -> None:
        s = summarize_portfolio([])
        assert s["count"] == 0

    def test_best_worst(self) -> None:
        items = [
            {"contract": {"symbol": "AAPL"}, "averageCost": 100,
             "marketPrice": 150, "marketValue": 15000,
             "unrealizedPNL": 5000, "realizedPNL": 0},
            {"contract": {"symbol": "MSFT"}, "averageCost": 400,
             "marketPrice": 350, "marketValue": 3500,
             "unrealizedPNL": -500, "realizedPNL": 100},
            {"contract": {"symbol": "NVDA"}, "averageCost": 800,
             "marketPrice": 820, "marketValue": 8200,
             "unrealizedPNL": 200, "realizedPNL": 50},
        ]
        s = summarize_portfolio(items)
        assert s["count"] == 3
        # AAPL: +50% (best); MSFT: -12.5% (worst).
        assert s["best_performer"]["symbol"] == "AAPL"
        assert s["worst_performer"]["symbol"] == "MSFT"


class TestSummarizeExecutions:
    def test_mixed_sides(self) -> None:
        trades = [
            {"execution": {"side": "BOT", "shares": 100, "avgPrice": 150},
             "commissionReport": {"commission": 1.0}},
            {"execution": {"side": "SLD", "shares": 50, "avgPrice": 155},
             "commissionReport": {"commission": 0.5}},
        ]
        s = summarize_executions(trades)
        assert s["count"] == 2
        assert s["total_commission"] == 1.5
        assert s["total_value"] == 100 * 150 + 50 * 155
