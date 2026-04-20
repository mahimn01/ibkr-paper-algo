"""Tests for the explain registry."""

from __future__ import annotations

import pytest

from trading_algo.explain import _EXPLANATIONS, all_explanations, explain


class TestShape:
    def test_every_entry_has_required_keys(self) -> None:
        required = {"action", "side_effects", "preconditions",
                    "reversibility", "idempotency"}
        for cmd, body in _EXPLANATIONS.items():
            missing = required - set(body.keys())
            assert not missing, f"{cmd} missing {missing}"

    def test_side_effects_is_list(self) -> None:
        for cmd, body in _EXPLANATIONS.items():
            assert isinstance(body["side_effects"], list), cmd

    def test_preconditions_is_list(self) -> None:
        for cmd, body in _EXPLANATIONS.items():
            assert isinstance(body["preconditions"], list), cmd


class TestCoverage:
    """High-risk commands must have real descriptions (not just fallback)."""
    @pytest.mark.parametrize("cmd", [
        "place-order", "place-bracket", "cancel-order", "cancel-all",
        "modify-order", "halt", "resume", "whatif",
        "history", "quote", "run",
    ])
    def test_critical_cmds_covered(self, cmd: str) -> None:
        assert cmd in _EXPLANATIONS

    def test_place_order_has_ibkr_notes(self) -> None:
        body = explain("place-order")
        assert "orderRef" in str(body).lower() or "orderRef" in "".join(body.get("notes", []))

    def test_history_mentions_pacing(self) -> None:
        body = explain("history")
        text = str(body).lower()
        assert "pacing" in text

    def test_halt_describes_sentinel(self) -> None:
        body = explain("halt")
        assert "HALTED" in body["action"] or "sentinel" in body["action"].lower()


class TestFallback:
    def test_unknown_falls_back(self) -> None:
        body = explain("no-such-cmd")
        assert "action" in body
        assert body["command"] == "no-such-cmd"


class TestAllExplanations:
    def test_returned_dict_is_copy(self) -> None:
        """Callers must not be able to mutate the registry."""
        out = all_explanations()
        out["_injected"] = "nope"
        assert "_injected" not in all_explanations()
