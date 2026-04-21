"""Tests for `tools-describe` JSONSchema generation."""

from __future__ import annotations

import json

import pytest

from trading_algo.tool_schema import describe_tools


@pytest.fixture
def cli_parser():
    from trading_algo.cli import build_parser
    return build_parser()


@pytest.fixture
def ibkr_parser():
    from trading_algo.ibkr_tool import build_parser
    return build_parser()


@pytest.fixture
def cli_tools(cli_parser):
    return describe_tools(cli_parser)


@pytest.fixture
def ibkr_tools(ibkr_parser):
    return describe_tools(ibkr_parser)


def _by_name(tools, name):
    for t in tools:
        if t["name"] == name:
            return t
    raise AssertionError(f"{name} not in tools")


class TestShape:
    def test_cli_non_empty(self, cli_tools) -> None:
        assert len(cli_tools) >= 10

    def test_ibkr_non_empty(self, ibkr_tools) -> None:
        assert len(ibkr_tools) >= 10

    def test_every_has_required(self, cli_tools, ibkr_tools) -> None:
        for tools in (cli_tools, ibkr_tools):
            for t in tools:
                assert "name" in t
                assert "description" in t
                assert "input_schema" in t
                assert "output_schema" in t

    def test_alphabetical(self, cli_tools) -> None:
        names = [t["name"] for t in cli_tools]
        assert names == sorted(names)


class TestPlaceOrderSchema:
    def test_required_fields(self, cli_tools) -> None:
        place = _by_name(cli_tools, "place-order")
        req = set(place["input_schema"].get("required", []))
        # `--type` has a default (MKT), so it's not argparse-required —
        # only the truly-required flags matter here.
        expected = {"symbol", "side", "qty"}
        assert expected.issubset(req)

    def test_idempotency_key_plumbed(self, cli_tools) -> None:
        place = _by_name(cli_tools, "place-order")
        props = place["input_schema"]["properties"]
        assert "idempotency_key" in props
        assert props["idempotency_key"]["type"] == "string"

    def test_kind_has_enum(self, cli_tools) -> None:
        place = _by_name(cli_tools, "place-order")
        kind = place["input_schema"]["properties"]["kind"]
        # cli.py place-order accepts the four placeable kinds (IND is
        # index, not directly placeable on most US venues).
        assert set(kind["enum"]) == {"STK", "OPT", "FUT", "FX"}

    def test_side_has_enum(self, cli_tools) -> None:
        place = _by_name(cli_tools, "place-order")
        side = place["input_schema"]["properties"]["side"]
        assert set(side["enum"]) == {"BUY", "SELL"}


class TestHaltResumeSchemas:
    def test_halt_requires_reason(self, cli_tools) -> None:
        halt = _by_name(cli_tools, "halt")
        assert "reason" in set(halt["input_schema"].get("required", []))

    def test_resume_has_confirm(self, cli_tools) -> None:
        resume = _by_name(cli_tools, "resume")
        props = resume["input_schema"]["properties"]
        assert "confirm_resume" in props
        assert props["confirm_resume"]["type"] == "boolean"


class TestOutputSchema:
    def test_envelope_shape(self, cli_tools) -> None:
        place = _by_name(cli_tools, "place-order")
        out = place["output_schema"]
        for k in ("ok", "cmd", "schema_version", "request_id",
                  "data", "warnings", "meta"):
            assert k in out["properties"]

    def test_cmd_const_pins_name(self, cli_tools) -> None:
        for t in cli_tools:
            assert t["output_schema"]["properties"]["cmd"]["const"] == t["name"]


class TestJsonSerialisable:
    def test_full_cli_to_json(self, cli_tools) -> None:
        text = json.dumps(cli_tools)
        assert json.loads(text) == cli_tools

    def test_full_ibkr_to_json(self, ibkr_tools) -> None:
        text = json.dumps(ibkr_tools)
        parsed = json.loads(text)
        assert len(parsed) == len(ibkr_tools)
