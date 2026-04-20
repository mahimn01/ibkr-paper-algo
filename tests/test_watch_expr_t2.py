"""Tests for the AST-restricted watch expression evaluator."""

from __future__ import annotations

import pytest

from trading_algo.watch_expr import UnsafeExpression, evaluate


class TestAllowedOperators:
    def test_simple_comparison(self) -> None:
        assert evaluate("last > 150", {"last": 151.0}) is True
        assert evaluate("last > 150", {"last": 149.0}) is False

    def test_chained_comparison(self) -> None:
        assert evaluate("100 < last < 200", {"last": 150.0}) is True
        assert evaluate("100 < last < 200", {"last": 250.0}) is False

    def test_equality(self) -> None:
        assert evaluate("status == 'Filled'", {"status": "Filled"}) is True
        assert evaluate("status != 'Filled'", {"status": "Filled"}) is False

    def test_logical_and_or(self) -> None:
        assert evaluate("last > 100 and vol > 1000", {"last": 150, "vol": 2000}) is True
        assert evaluate("last > 100 or vol > 1000", {"last": 50, "vol": 2000}) is True
        assert evaluate("last > 100 and vol > 1000", {"last": 150, "vol": 500}) is False

    def test_not(self) -> None:
        assert evaluate("not filled", {"filled": False}) is True
        assert evaluate("not filled", {"filled": True}) is False

    def test_arithmetic(self) -> None:
        assert evaluate("bid + ask > 300", {"bid": 150.0, "ask": 151.0}) is True
        assert evaluate("(bid + ask) / 2 > 150", {"bid": 150.0, "ask": 151.0}) is True

    def test_unary_minus(self) -> None:
        assert evaluate("-pnl > 100", {"pnl": -200}) is True

    def test_bool_and_none_literals(self) -> None:
        assert evaluate("True", {}) is True
        assert evaluate("False", {}) is False
        assert evaluate("flag == None", {"flag": None}) is True


class TestMissingNames:
    def test_missing_name_resolves_to_none(self) -> None:
        # Comparison with None → False (except ==, !=).
        assert evaluate("missing > 5", {}) is False
        assert evaluate("missing == None", {}) is True
        assert evaluate("missing != None", {"missing": 1}) is True


class TestDisallowedConstructs:
    def test_function_call_rejected(self) -> None:
        with pytest.raises(UnsafeExpression):
            evaluate("abs(last)", {"last": -5})

    def test_attribute_access_rejected(self) -> None:
        with pytest.raises(UnsafeExpression):
            evaluate("quote.last > 5", {"quote": object()})

    def test_subscript_rejected(self) -> None:
        with pytest.raises(UnsafeExpression):
            evaluate("data[0] > 5", {"data": [1, 2, 3]})

    def test_lambda_rejected(self) -> None:
        with pytest.raises(UnsafeExpression):
            evaluate("(lambda: 1)()", {})

    def test_list_comp_rejected(self) -> None:
        with pytest.raises(UnsafeExpression):
            evaluate("[x for x in [1,2,3]]", {})

    def test_dict_rejected(self) -> None:
        with pytest.raises(UnsafeExpression):
            evaluate("{'a': 1}", {})

    def test_import_rejected_at_parse(self) -> None:
        # import is a statement, not an expression — parser rejects first.
        with pytest.raises((UnsafeExpression, SyntaxError)):
            evaluate("import os", {})

    def test_bitwise_rejected(self) -> None:
        with pytest.raises(UnsafeExpression):
            evaluate("flags & 1", {"flags": 3})


class TestEdgeCases:
    def test_empty_expression_errors(self) -> None:
        with pytest.raises(ValueError):
            evaluate("", {})
        with pytest.raises(ValueError):
            evaluate("   ", {})

    def test_syntax_error_propagates(self) -> None:
        with pytest.raises(SyntaxError):
            evaluate("last >", {"last": 1})

    def test_string_comparison(self) -> None:
        assert evaluate("side == 'BUY'", {"side": "BUY"}) is True
        assert evaluate("side == 'BUY'", {"side": "SELL"}) is False

    def test_chained_all_must_hold(self) -> None:
        # Classic chained: 0 < x < 100 and x != 50
        env = {"x": 75}
        assert evaluate("0 < x < 100 and x != 50", env) is True
        env["x"] = 50
        assert evaluate("0 < x < 100 and x != 50", env) is False
