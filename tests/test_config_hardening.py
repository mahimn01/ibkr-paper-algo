"""Tests for the hardened env-bool parser + atomic_write_text.

These cover the Wave T1 safety fix: a typo on `TRADING_ALLOW_LIVE` can no
longer silently flip the gate off — it raises `EnvParseError` at startup.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from trading_algo.config import (
    EnvParseError,
    TradingConfig,
    _get_env_bool,
    _get_env_float,
    _get_env_int,
    atomic_write_text,
)


# -----------------------------------------------------------------------------
# Strict env-bool
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("TRUE", True), ("True", True),
    ("1", True), ("yes", True), ("YES", True), ("on", True), ("y", True),
    ("false", False), ("FALSE", False), ("False", False),
    ("0", False), ("no", False), ("NO", False), ("off", False), ("n", False),
    ("", False),  # empty string treated as falsy (unset-like)
    ("  true  ", True), ("  FALSE  ", False),  # whitespace tolerated
])
def test_env_bool_accepts_canonical(monkeypatch, raw, expected):
    monkeypatch.setenv("TEST_FLAG", raw)
    assert _get_env_bool("TEST_FLAG", default=not expected) is expected


def test_env_bool_unset_returns_default(monkeypatch):
    monkeypatch.delenv("TEST_FLAG", raising=False)
    assert _get_env_bool("TEST_FLAG", default=True) is True
    assert _get_env_bool("TEST_FLAG", default=False) is False


@pytest.mark.parametrize("garbage", [
    "flase",   # typo
    "tru",     # typo
    "YEP",     # non-canonical
    "enable",
    "disable",
    "TRUE!",
    "2",
    "-1",
])
def test_env_bool_rejects_garbage(monkeypatch, garbage):
    """Safety: a typo on a critical flag must never silently default."""
    monkeypatch.setenv("TEST_FLAG", garbage)
    with pytest.raises(EnvParseError):
        _get_env_bool("TEST_FLAG", default=False)


def test_env_int_rejects(monkeypatch):
    monkeypatch.setenv("N", "not-a-number")
    with pytest.raises(EnvParseError):
        _get_env_int("N", 0)


def test_env_float_rejects(monkeypatch):
    monkeypatch.setenv("X", "3.14.15")
    with pytest.raises(EnvParseError):
        _get_env_float("X", 0.0)


def test_trading_config_typo_on_live_raises(monkeypatch):
    """The exact bug this fix targets: `TRADING_ALLOW_LIVE=tru` (typo)
    no longer silently keeps live-trading disabled — it crashes at
    startup, forcing the operator to fix the config.
    """
    monkeypatch.setenv("TRADING_ALLOW_LIVE", "tru")
    with pytest.raises(EnvParseError):
        TradingConfig.from_env()


def test_trading_config_mixed_case_accepted(monkeypatch):
    monkeypatch.setenv("TRADING_ALLOW_LIVE", "TRUE")
    monkeypatch.setenv("TRADING_LIVE_ENABLED", "True")
    monkeypatch.setenv("TRADING_DRY_RUN", "FALSE")
    cfg = TradingConfig.from_env()
    assert cfg.allow_live is True
    assert cfg.live_enabled is True
    assert cfg.dry_run is False


# -----------------------------------------------------------------------------
# atomic_write_text
# -----------------------------------------------------------------------------

def test_atomic_write_creates_file(tmp_path):
    p = tmp_path / "out.txt"
    atomic_write_text(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"


def test_atomic_write_overwrites(tmp_path):
    p = tmp_path / "out.txt"
    atomic_write_text(p, "first")
    atomic_write_text(p, "second")
    assert p.read_text(encoding="utf-8") == "second"


def test_atomic_write_creates_parents(tmp_path):
    p = tmp_path / "a" / "b" / "c" / "out.txt"
    atomic_write_text(p, "hi")
    assert p.read_text(encoding="utf-8") == "hi"


def test_atomic_write_mode_0600(tmp_path):
    p = tmp_path / "out.txt"
    atomic_write_text(p, "hi", mode=0o600)
    if os.name == "posix":
        assert p.stat().st_mode & 0o777 == 0o600


def test_atomic_write_no_leftover_tempfiles(tmp_path):
    p = tmp_path / "out.txt"
    for i in range(10):
        atomic_write_text(p, str(i))
    leftovers = [x for x in tmp_path.iterdir()
                 if x.name.startswith(".out.txt.") and x.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_write_concurrent_no_corruption(tmp_path):
    p = tmp_path / "out.txt"

    def w(i: int) -> None:
        for j in range(20):
            atomic_write_text(p, f"thread-{i}-iter-{j}")

    threads = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Whatever the winning write, the file must be non-empty + well-formed.
    content = p.read_text(encoding="utf-8")
    assert content.startswith("thread-")
