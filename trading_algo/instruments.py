from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


InstrumentKind = Literal["STK", "FUT", "FX", "OPT"]


_FUT_EXPIRY_RE = re.compile(r"^\d{6}(\d{2})?$")  # YYYYMM or YYYYMMDD
_FX_PAIR_RE = re.compile(r"^[A-Z]{6}$")  # e.g. EURUSD
_OPT_EXPIRY_RE = re.compile(r"^\d{8}$")  # YYYYMMDD


@dataclass(frozen=True)
class InstrumentSpec:
    """
    Canonical instrument description used throughout the system.

    - STK: symbol=IBM exchange=SMART currency=USD
    - FUT: symbol=ES exchange=CME expiry=202503 currency=USD
    - FX:  symbol=EURUSD exchange=IDEALPRO
    - OPT: symbol=AAPL expiry=20260116 right=C strike=200 exchange=SMART currency=USD multiplier=100
    """

    kind: InstrumentKind
    symbol: str
    exchange: str | None = None
    currency: str | None = None
    expiry: str | None = None
    right: str | None = None  # OPT only: C|P
    strike: float | None = None  # OPT only
    multiplier: str | None = None  # OPT only (often "100")

    def normalized(self) -> "InstrumentSpec":
        kind = self.kind.upper()
        symbol = self.symbol.upper()
        exchange = (self.exchange or "").upper() or None
        currency = (self.currency or "").upper() or None
        expiry = self.expiry
        right = (self.right or "").upper() or None
        strike = float(self.strike) if self.strike is not None else None
        multiplier = str(self.multiplier).strip() if self.multiplier is not None and str(self.multiplier).strip() else None
        return InstrumentSpec(
            kind=kind,
            symbol=symbol,
            exchange=exchange,
            currency=currency,
            expiry=expiry,
            right=right,
            strike=strike,
            multiplier=multiplier,
        )


def validate_instrument(spec: InstrumentSpec) -> InstrumentSpec:
    spec = spec.normalized()

    if spec.kind not in {"STK", "FUT", "FX", "OPT"}:
        raise ValueError(f"Unsupported instrument kind: {spec.kind}")
    if not spec.symbol:
        raise ValueError("Instrument symbol is required")

    if spec.kind == "STK":
        exchange = spec.exchange or "SMART"
        currency = spec.currency or "USD"
        return InstrumentSpec(kind="STK", symbol=spec.symbol, exchange=exchange, currency=currency)

    if spec.kind == "FUT":
        if not spec.exchange:
            raise ValueError("FUT exchange is required (e.g. CME)")
        if not spec.expiry or not _FUT_EXPIRY_RE.match(spec.expiry):
            raise ValueError("FUT expiry must be YYYYMM or YYYYMMDD (e.g. 202503 or 20250315)")
        currency = spec.currency or "USD"
        return InstrumentSpec(kind="FUT", symbol=spec.symbol, exchange=spec.exchange, currency=currency, expiry=spec.expiry)

    if spec.kind == "OPT":
        exchange = spec.exchange or "SMART"
        currency = spec.currency or "USD"
        if not spec.expiry or not _OPT_EXPIRY_RE.match(spec.expiry):
            raise ValueError("OPT expiry must be YYYYMMDD (e.g. 20260116)")
        right = (spec.right or "").upper()
        if right not in {"C", "P"}:
            raise ValueError("OPT right must be C or P")
        if spec.strike is None or float(spec.strike) <= 0:
            raise ValueError("OPT strike must be positive")
        multiplier = spec.multiplier or "100"
        return InstrumentSpec(
            kind="OPT",
            symbol=spec.symbol,
            exchange=exchange,
            currency=currency,
            expiry=spec.expiry,
            right=right,
            strike=float(spec.strike),
            multiplier=multiplier,
        )

    # FX
    exchange = spec.exchange or "IDEALPRO"
    if not _FX_PAIR_RE.match(spec.symbol):
        raise ValueError("FX symbol must be a 6-letter pair like EURUSD")
    return InstrumentSpec(kind="FX", symbol=spec.symbol, exchange=exchange)
