"""Tests for T3.4 --confirm-panic gate on cancel-all."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from trading_algo import ibkr_tool


def _args(**kw) -> argparse.Namespace:
    defaults = {
        "yes": False,
        "confirm_panic": False,
        "format": "json",
        "host": None,
        "port": None,
        "client_id": None,
        "paper": True,
    }
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestCancelAllGates:
    def test_refuses_without_yes(self) -> None:
        with pytest.raises(SystemExit) as exc:
            ibkr_tool.cmd_cancel_all(_args(yes=False))
        assert "without --yes" in str(exc.value)

    def test_refuses_with_yes_but_no_panic(self) -> None:
        with pytest.raises(SystemExit) as exc:
            ibkr_tool.cmd_cancel_all(_args(yes=True, confirm_panic=False))
        assert "--confirm-panic" in str(exc.value)

    def test_accepts_with_both(self, monkeypatch) -> None:
        fake_ib = MagicMock()
        monkeypatch.setattr(ibkr_tool, "_connect", lambda *a, **k: fake_ib)

        rc = ibkr_tool.cmd_cancel_all(_args(yes=True, confirm_panic=True))
        assert rc == 0
        fake_ib.reqGlobalCancel.assert_called_once()
        fake_ib.disconnect.assert_called_once()

    def test_halt_still_blocks(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("TRADING_HALT_PATH", str(tmp_path / "HALTED"))
        from trading_algo.halt import write_halt, HaltActive
        write_halt(reason="maintenance", by="ops")

        with pytest.raises(HaltActive):
            ibkr_tool.cmd_cancel_all(_args(yes=True, confirm_panic=True))


class TestParserRegisters:
    def test_flag_registered(self) -> None:
        p = ibkr_tool.build_parser()
        sub_action = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        ca = sub_action.choices["cancel-all"]
        flags = {a.dest for a in ca._actions}
        assert "confirm_panic" in flags
