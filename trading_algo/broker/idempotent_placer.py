"""Idempotent order placement wrapper for IBKR.

Problem: IBKR has no strong server-side dedup on `placeOrder`. A naive retry
after a transient error (TWS disconnect, timeout, pacing) will re-transmit
the order — one agent crash becomes two fills.

Solution: pattern borrowed from Stripe's `Idempotency-Key` + Kite's
`tag`-based dedup, adapted for IBKR's `orderRef` field (free-text,
travels with the order through the exchange, reported back in open/
completed trades):

1. Derive a deterministic `orderRef` from the caller's idempotency key
   via BLAKE2b (see `trading_algo.idempotency.derive_order_ref`).
2. Before placing, call `ib.openTrades()` + `ib.reqCompletedOrders()`
   and match on `orderRef`. If found → return existing Trade's state.
   Never re-transmit.
3. If the orderbook lookup ITSELF fails (not just "not found" — the
   underlying IB call raised), raise `IBKROrderbookLookupError`. Do
   NOT retry placeOrder: we can't tell whether the order landed and a
   blind retry is the exact double-fill bug we're defending against.
4. On transient place-side errors, re-check the orderbook before
   retrying. Same idea: confirm the order didn't already reach IBKR.

IBKR `orderRef` notes (from official docs + forum):
- Free-text, up to ~40 chars. Longer values are truncated by some
  venues but accepted at the API layer.
- Included in `Order.orderRef` and reported back in:
  - `ib.openTrades()` → `Trade.order.orderRef`
  - `ib.reqCompletedOrders(apiOnly=False)` → same structure
  - Flex Web Service activity statements (`OrderReference` column)
- Survives TWS restart — part of the order record, not client state.
- NOT unique-enforced by IBKR. If you reuse an orderRef across
  multiple live orders, the API accepts all of them — THIS wrapper
  is what enforces uniqueness by querying first.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from trading_algo.broker.base import OrderRequest, OrderResult

log = logging.getLogger(__name__)


class IBKROrderbookLookupError(RuntimeError):
    """Raised when we cannot read the orderbook to check for a placed order.

    Critical distinction from "order not found": if the orderbook is
    unreachable, we genuinely do not know whether our prior attempt
    landed. Blindly retrying `placeOrder` in that state risks a double-
    fill. The safe behaviour is to surface this error and let the caller
    decide (reconcile later, alert human, etc).
    """


# ---------------------------------------------------------------------------
# Orderbook lookup
# ---------------------------------------------------------------------------

# Classes for error-classification. We match by class name to avoid a hard
# import dependency on ib_async at module-load time.
_HARD_ERROR_NAMES = frozenset({
    "InputException", "OrderException", "PermissionException",
    "KeyboardInterrupt", "SystemExit", "ValueError", "TypeError",
    "IBKRDependencyError",
})

_TRANSIENT_MARKERS = (
    "timeout", "timed out", "connection reset", "temporarily unavailable",
    "500", "502", "503", "504",
    "rate limit", "too many requests", "pacing",
)


def _is_transient(exc: BaseException) -> bool:
    """Is this a transient (retryable) error?"""
    name = type(exc).__name__
    if name in _HARD_ERROR_NAMES:
        return False
    if name in ("ConnectionError", "TimeoutError"):
        return True
    msg = str(exc).lower()
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return True
    # IBKR errorCode 1100/162 are transient.
    code = getattr(exc, "errorCode", None) or getattr(exc, "code", None)
    if isinstance(code, int) and code in (1100, 1101, 1102, 162, 165, 420):
        return True
    return False


def find_trade_by_order_ref(ib: Any, order_ref: str) -> Any | None:
    """Search IBKR's orderbook (open + completed today) for a Trade whose
    `order.orderRef == order_ref`.

    Returns:
        - The matching Trade object (happy path).
        - `None` — orderbook fetched successfully, no match.

    Raises:
        `IBKROrderbookLookupError` — the underlying IB call(s) failed.
        Callers must NOT interpret as "not found" — a blind retry of
        placeOrder in this state is the double-fill bug.
    """
    if ib is None:
        raise IBKROrderbookLookupError("ib client is None")

    # Case-insensitive compare — some flows normalise case.
    wanted = (order_ref or "").upper()

    # 1. Open trades (cross-client view; requires client_id=0 or master).
    try:
        open_trades = list(ib.openTrades() or [])
    except Exception as exc:
        raise IBKROrderbookLookupError(
            f"openTrades() failed: {type(exc).__name__}: {exc}"
        ) from exc

    for t in open_trades:
        order = getattr(t, "order", None)
        if order is None:
            continue
        ref = (getattr(order, "orderRef", "") or "").upper()
        if ref == wanted:
            return t

    # 2. Completed orders (today's terminal orders).
    try:
        completed = list(ib.reqCompletedOrders(apiOnly=False) or [])
    except TypeError:
        # Older ib_async signatures don't accept apiOnly — try without.
        try:
            completed = list(ib.reqCompletedOrders() or [])
        except Exception as exc:
            raise IBKROrderbookLookupError(
                f"reqCompletedOrders() failed: {type(exc).__name__}: {exc}"
            ) from exc
    except Exception as exc:
        raise IBKROrderbookLookupError(
            f"reqCompletedOrders() failed: {type(exc).__name__}: {exc}"
        ) from exc

    for t in completed:
        order = getattr(t, "order", None)
        if order is None:
            continue
        ref = (getattr(order, "orderRef", "") or "").upper()
        if ref == wanted:
            return t

    return None


# ---------------------------------------------------------------------------
# Trade → OrderResult
# ---------------------------------------------------------------------------

def _trade_to_result(trade: Any) -> OrderResult:
    """Extract an OrderResult from an ib_async Trade."""
    order = getattr(trade, "order", None)
    status = getattr(getattr(trade, "orderStatus", None), "status", None)
    order_id = getattr(order, "orderId", None) if order is not None else None
    return OrderResult(
        order_id=str(order_id) if order_id is not None else "unknown",
        status=str(status or "unknown"),
    )


# ---------------------------------------------------------------------------
# Placer
# ---------------------------------------------------------------------------

# Poll delays AFTER a transient error: give IBKR's OMS up to ~7.5s to reflect
# the order. Longer than 7.5s → give up and raise.
_ORDERBOOK_POLL_DELAYS = (0.5, 1.0, 1.5, 2.0, 2.5)


@dataclass
class IdempotentOrderPlacer:
    """Wrap a broker-like object's `place_order` with IBKR orderRef-based
    idempotency.

    Usage:
        broker = IBKRBroker(...)
        placer = IdempotentOrderPlacer(broker)
        result = placer.place(
            req,                                 # OrderRequest
            idempotency_key="agent-turn-42",     # stable across retries
        )
    """
    broker: Any                                  # IBKRBroker (or duck-compatible)
    max_attempts: int = 3
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 10.0

    def place(
        self,
        req: OrderRequest,
        *,
        idempotency_key: str | None = None,
        order_ref: str | None = None,
    ) -> OrderResult:
        """Idempotent place.

        If `order_ref` is already set on `req`, use it verbatim.
        Else if `idempotency_key` is provided, derive `order_ref` from it
        via BLAKE2b.
        If neither — auto-mint a random order_ref (NOT idempotent across
        process restarts, but at least traceable in Flex statements).
        """
        effective_ref = self._resolve_order_ref(req, idempotency_key, order_ref)
        req_with_ref = _with_order_ref(req, effective_ref)

        ib = self._ib()

        # 1. Pre-check: is this order already known to IBKR?
        existing = find_trade_by_order_ref(ib, effective_ref)
        if existing is not None:
            log.info(
                "idempotent-place: orderRef=%s already exists on IBKR — replaying",
                effective_ref,
            )
            return _trade_to_result(existing)

        # 2. Not found. Place it.
        delay = self.initial_backoff_s
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.broker.place_order(req_with_ref)
                log.info(
                    "idempotent-place: submitted orderRef=%s order_id=%s status=%s",
                    effective_ref, result.order_id, result.status,
                )
                return result
            except Exception as exc:
                last_exc = exc
                name = type(exc).__name__
                if name in _HARD_ERROR_NAMES:
                    log.error("idempotent-place: hard error %s — not retrying: %s",
                              name, exc)
                    raise
                if not _is_transient(exc):
                    raise

                # Transient — before retrying, query the orderbook. If the
                # order DID land despite the error, return its state.
                log.warning(
                    "idempotent-place: transient error on attempt %d/%d: %s; "
                    "polling orderbook for orderRef=%s",
                    attempt, self.max_attempts, exc, effective_ref,
                )
                found = self._poll_orderbook(ib, effective_ref)
                if found is not None:
                    log.info(
                        "idempotent-place: order appeared on IBKR after "
                        "transient error — orderRef=%s status=%s",
                        effective_ref,
                        getattr(getattr(found, "orderStatus", None), "status", "?"),
                    )
                    return _trade_to_result(found)

                if attempt == self.max_attempts:
                    break

                sleep_for = min(self.max_backoff_s, delay) * (0.5 + random.random())
                log.info("idempotent-place: retry in %.2fs", sleep_for)
                time.sleep(sleep_for)
                delay *= 2

        assert last_exc is not None
        raise last_exc

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _ib(self) -> Any:
        """Reach into the wrapped broker for the ib_async client.

        We support two shapes: `broker._ib` (trading-algo's IBKRBroker) or
        `broker.ib` (alternate). Raise if neither exists — we can't do
        orderbook lookups without an IB client.
        """
        ib = getattr(self.broker, "_ib", None) or getattr(self.broker, "ib", None)
        if ib is None:
            raise RuntimeError(
                "IdempotentOrderPlacer: wrapped broker has no `_ib` / `ib` "
                "attribute. Broker must be connected."
            )
        return ib

    def _resolve_order_ref(
        self,
        req: OrderRequest,
        idempotency_key: str | None,
        order_ref: str | None,
    ) -> str:
        """Precedence: explicit order_ref > req.order_ref > derived > random."""
        from trading_algo.idempotency import derive_order_ref

        if order_ref:
            return order_ref
        if getattr(req, "order_ref", None):
            return req.order_ref  # type: ignore[return-value]
        if idempotency_key:
            return derive_order_ref(idempotency_key)
        # No key — mint a random ref. Logged but NOT idempotent across
        # process restarts. Callers that care about crash safety MUST
        # pass an idempotency_key.
        import secrets
        ref = f"TA{secrets.token_hex(14).upper()}"
        log.warning(
            "idempotent-place: no idempotency_key supplied; using ephemeral "
            "orderRef=%s. Crash recovery will NOT find this order.", ref,
        )
        return ref

    def _poll_orderbook(self, ib: Any, order_ref: str) -> Any | None:
        """Poll the orderbook with increasing delays to tolerate OMS lag.

        Returns Trade if found, None if definitely not.
        Raises IBKROrderbookLookupError only if *every* lookup in the
        poll window failed (can't verify → must NOT retry placeOrder).
        """
        successes = 0
        last_err: IBKROrderbookLookupError | None = None
        for step in _ORDERBOOK_POLL_DELAYS:
            time.sleep(step)
            try:
                found = find_trade_by_order_ref(ib, order_ref)
            except IBKROrderbookLookupError as exc:
                last_err = exc
                continue
            successes += 1
            if found is not None:
                return found
        if successes == 0 and last_err is not None:
            raise last_err
        return None


# ---------------------------------------------------------------------------
# OrderRequest helper
# ---------------------------------------------------------------------------

def _with_order_ref(req: OrderRequest, order_ref: str) -> OrderRequest:
    """Return a copy of `req` with `order_ref` populated.

    OrderRequest is a frozen dataclass — we replace() instead of mutate.
    """
    from dataclasses import replace
    return replace(req, order_ref=order_ref)
