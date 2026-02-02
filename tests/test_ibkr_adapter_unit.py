import logging
import unittest

from trading_algo.broker.base import OrderRequest
from trading_algo.broker.ibkr import IBKRBroker, _Factories
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec

logging.disable(logging.CRITICAL)


class _FakeTrade:
    def __init__(self, order_id: int, status: str):
        class _Order:
            def __init__(self, oid):
                self.orderId = oid

        class _OrderStatus:
            def __init__(self, st):
                self.status = st

        self.order = _Order(order_id)
        self.orderStatus = _OrderStatus(status)


class _FakeIB:
    def __init__(self):
        self.calls = []
        self._accounts = ["DU12345"]

    def connect(self, host, port, clientId):
        self.calls.append(("connect", host, port, clientId))

    def disconnect(self):
        self.calls.append(("disconnect",))

    def qualifyContracts(self, contract):
        self.calls.append(("qualify", contract))
        return [contract]

    def placeOrder(self, contract, order):
        self.calls.append(("placeOrder", contract, order))
        return _FakeTrade(123, "Submitted")

    def sleep(self, _seconds):
        self.calls.append(("sleep",))

    def reqMktData(self, contract, *_args):
        self.calls.append(("reqMktData", contract))

        class _Ticker:
            bid = 1.0
            ask = 2.0
            last = 1.5
            close = 1.4
            volume = 10

        return _Ticker()

    def managedAccounts(self):
        return list(self._accounts)

    def accountSummary(self, _account):
        return []

    def positions(self):
        return []

    def reqHistoricalData(self, contract, **kwargs):
        self.calls.append(("reqHistoricalData", contract, kwargs))

        class _Bar:
            def __init__(self):
                self.date = None
                self.open = 1
                self.high = 2
                self.low = 0.5
                self.close = 1.5
                self.volume = 10

        return [_Bar(), _Bar()]


class _FakeStock:
    def __init__(self, symbol, exchange, currency):
        self.kind = "STK"
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency


class _FakeFuture:
    def __init__(self, symbol, expiry, exchange, currency=None):
        self.kind = "FUT"
        self.symbol = symbol
        self.expiry = expiry
        self.exchange = exchange
        self.currency = currency


class _FakeForex:
    def __init__(self, pair):
        self.kind = "FX"
        self.pair = pair


class _FakeOption:
    def __init__(self, symbol, expiry, strike, right, exchange, currency=None):
        self.kind = "OPT"
        self.symbol = symbol
        self.lastTradeDateOrContractMonth = expiry
        self.strike = strike
        self.right = right
        self.exchange = exchange
        self.currency = currency
        self.multiplier = None


class _FakeMarketOrder:
    def __init__(self, side, qty, tif=None):
        self.side = side
        self.qty = qty
        self.tif = tif
        self.orderId = None
        self.outsideRth = False
        self.transmit = True


class _FakeLimitOrder:
    def __init__(self, side, qty, price, tif=None):
        self.side = side
        self.qty = qty
        self.price = price
        self.tif = tif
        self.orderId = None
        self.outsideRth = False
        self.transmit = True


class _FakeStopOrder:
    def __init__(self, side, qty, stop_price, tif=None):
        self.side = side
        self.qty = qty
        self.stop_price = stop_price
        self.tif = tif
        self.orderId = None
        self.outsideRth = False
        self.transmit = True


class _FakeStopLimitOrder:
    def __init__(self, side, qty, limit_price, stop_price, tif=None):
        self.side = side
        self.qty = qty
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.tif = tif
        self.orderId = None
        self.outsideRth = False
        self.transmit = True


class TestIBKRAdapterUnit(unittest.TestCase):
    def _make_broker(self):
        broker = IBKRBroker(config=IBKRConfig(), ib_factory=_FakeIB)
        broker._factories = _Factories(
            IB=_FakeIB,
            Stock=_FakeStock,
            Future=_FakeFuture,
            Option=_FakeOption,
            Forex=_FakeForex,
            MarketOrder=_FakeMarketOrder,
            LimitOrder=_FakeLimitOrder,
            StopOrder=_FakeStopOrder,
            StopLimitOrder=_FakeStopLimitOrder,
        )
        broker.connect()
        self.addCleanup(broker.disconnect)
        return broker

    def test_places_stock_order_with_qualification(self):
        broker = self._make_broker()
        req = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1, order_type="MKT")
        result = broker.place_order(req)
        self.assertEqual(result.order_id, "123")

        calls = broker._ib.calls
        self.assertEqual(calls[0][0], "connect")
        self.assertEqual(calls[1][0], "qualify")
        self.assertEqual(calls[2][0], "placeOrder")

        _, contract = calls[1]
        self.assertEqual(contract.kind, "STK")
        self.assertEqual(contract.symbol, "AAPL")
        self.assertEqual(contract.exchange, "SMART")
        self.assertEqual(contract.currency, "USD")

    def test_contract_mapping_future_and_fx(self):
        broker = self._make_broker()

        fut_req = OrderRequest(
            instrument=InstrumentSpec(kind="FUT", symbol="ES", exchange="CME", expiry="202503"),
            side="BUY",
            quantity=1,
            order_type="MKT",
        )
        broker.place_order(fut_req)
        # Find the qualify call for the futures contract
        qualify_calls = [c for c in broker._ib.calls if c[0] == "qualify"]
        self.assertGreaterEqual(len(qualify_calls), 1)
        _, fut_contract = qualify_calls[-1]
        self.assertEqual(fut_contract.kind, "FUT")
        self.assertEqual(fut_contract.symbol, "ES")
        self.assertEqual(fut_contract.exchange, "CME")
        self.assertEqual(fut_contract.expiry, "202503")

        fx_req = OrderRequest(instrument=InstrumentSpec(kind="FX", symbol="EURUSD"), side="BUY", quantity=1, order_type="MKT")
        broker.place_order(fx_req)
        # Find the qualify call for the FX contract
        qualify_calls = [c for c in broker._ib.calls if c[0] == "qualify"]
        self.assertGreaterEqual(len(qualify_calls), 2)
        _, fx_contract = qualify_calls[-1]
        self.assertEqual(fx_contract.kind, "FX")
        self.assertEqual(fx_contract.pair, "EURUSD")

    def test_contract_mapping_option(self):
        broker = self._make_broker()
        opt_req = OrderRequest(
            instrument=InstrumentSpec(kind="OPT", symbol="AAPL", expiry="20260116", right="C", strike=200),
            side="BUY",
            quantity=1,
            order_type="MKT",
        )
        broker.place_order(opt_req)
        # Find the qualify call for the option contract
        qualify_calls = [c for c in broker._ib.calls if c[0] == "qualify"]
        self.assertGreaterEqual(len(qualify_calls), 1)
        _, opt_contract = qualify_calls[-1]
        self.assertEqual(opt_contract.kind, "OPT")
        self.assertEqual(opt_contract.symbol, "AAPL")
        self.assertEqual(opt_contract.lastTradeDateOrContractMonth, "20260116")
        self.assertEqual(opt_contract.right, "C")
        self.assertEqual(opt_contract.strike, 200.0)
        self.assertEqual(opt_contract.exchange, "SMART")
        self.assertEqual(opt_contract.currency, "USD")

    def test_snapshot_calls_req_mkt_data(self):
        broker = self._make_broker()
        snap = broker.get_market_data_snapshot(InstrumentSpec(kind="STK", symbol="AAPL"))
        self.assertEqual(snap.bid, 1.0)
        self.assertEqual(snap.ask, 2.0)
        self.assertEqual(snap.last, 1.5)
        self.assertEqual(snap.instrument.symbol, "AAPL")

    def test_history_calls_req_historical_data(self):
        broker = self._make_broker()
        bars = broker.get_historical_bars(InstrumentSpec(kind="STK", symbol="AAPL"), end_datetime=None, duration="1 D", bar_size="5 mins")
        self.assertEqual(len(bars), 2)
        calls = [c for c in broker._ib.calls if c[0] == "reqHistoricalData"]
        self.assertEqual(len(calls), 1)

    def test_modify_calls_place_order_with_order_id(self):
        broker = self._make_broker()
        req = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1, order_type="LMT", limit_price=100)
        res = broker.place_order(req)
        new_req = OrderRequest(instrument=InstrumentSpec(kind="STK", symbol="AAPL"), side="BUY", quantity=1, order_type="LMT", limit_price=99)
        broker.modify_order(res.order_id, new_req)
        # find second placeOrder call
        place_calls = [c for c in broker._ib.calls if c[0] == "placeOrder"]
        self.assertGreaterEqual(len(place_calls), 2)
        _, _contract, order = place_calls[-1]
        self.assertEqual(order.orderId, int(res.order_id))
