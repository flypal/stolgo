from stolgo.core.types import Bar, OrderType, Side
from stolgo.oms.commission import BpsCommission
from stolgo.oms.fill_model import NextOpenFill
from stolgo.oms.sim_broker import SimBroker
from stolgo.oms.slippage import NoSlippage


def test_market_fill_next_open():
    broker = SimBroker(NextOpenFill(), NoSlippage(), BpsCommission(0.0))
    order = broker.create_order("X", Side.BUY, 1.0, OrderType.MARKET)
    broker.submit(order)
    bar = Bar(ts=1, open=10.0, high=11, low=9, close=10.5, volume=1, symbol="X")
    fills = broker.match(bar, bar_index=1)
    assert len(fills) == 1
    assert fills[0].fill.price == 10.0


def _stop_fill_price(side, stop, bar):
    broker = SimBroker(NextOpenFill(), NoSlippage(), BpsCommission(0.0))
    broker.submit(broker.create_order("X", side, 1.0, OrderType.STOP, stop_price=stop))
    fills = broker.match(bar, bar_index=1)
    return fills[0].fill.price if fills else None


def test_stop_fills_at_trigger_or_gap_open():
    bar = Bar(ts=1, open=10.0, high=12, low=9, close=11, volume=1, symbol="X")
    assert _stop_fill_price(Side.BUY, 11.0, bar) == 11.0
    assert _stop_fill_price(Side.SELL, 9.5, bar) == 9.5
    assert _stop_fill_price(Side.BUY, 13.0, bar) is None
    gap_up = Bar(ts=1, open=12.0, high=13, low=11.5, close=12.5, volume=1, symbol="X")
    assert _stop_fill_price(Side.BUY, 11.0, gap_up) == 12.0


def _bar(o, h, l, c, ts=1):
    return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=1, symbol="X")


def _broker():
    return SimBroker(NextOpenFill(), NoSlippage(), BpsCommission(0.0))


def _fills(broker, bar, i):
    return [(fe.fill.side, fe.fill.price) for fe in broker.match(bar, bar_index=i)]


def test_order_expires_after_its_last_valid_bar():
    broker = _broker()
    broker.submit(broker.create_order("X", Side.BUY, 1.0, OrderType.STOP, stop_price=11.0, expire_index=1))
    assert _fills(broker, _bar(10, 10.5, 9, 10), 1) == []
    assert broker.open_orders() == []
    assert _fills(broker, _bar(10, 12, 9, 11), 2) == []


def _long_bracket(broker, **kw):
    broker.submit(
        broker.create_order(
            "X", Side.BUY, 1.0, OrderType.STOP, stop_price=11.0, stop_loss=9.0, take_profit=13.0, **kw
        )
    )


def test_bracket_legs_rest_after_fill_and_target_cancels_stop():
    broker = _broker()
    _long_bracket(broker)
    assert _fills(broker, _bar(10, 11.5, 10, 11), 1) == [(Side.BUY, 11.0)]
    assert sorted(o.order_type for o in broker.open_orders()) == [OrderType.LIMIT, OrderType.STOP]
    assert _fills(broker, _bar(12, 13.5, 11.5, 13), 2) == [(Side.SELL, 13.0)]
    assert broker.open_orders() == []


def test_stop_leg_can_fill_on_entry_bar_at_stop_price():
    broker = _broker()
    _long_bracket(broker)
    # opens below the stop-loss, but that print came before the entry fill
    assert _fills(broker, _bar(8.5, 11.5, 8.5, 9), 1) == [(Side.BUY, 11.0), (Side.SELL, 9.0)]


def test_bar_touching_both_legs_assumes_stop_unless_open_is_past_target():
    broker = _broker()
    _long_bracket(broker)
    _fills(broker, _bar(10, 11.5, 10, 11), 1)
    assert _fills(broker, _bar(11, 14, 8, 11), 2) == [(Side.SELL, 9.0)]

    broker = _broker()
    _long_bracket(broker)
    _fills(broker, _bar(10, 11.5, 10, 11), 1)
    assert _fills(broker, _bar(13.5, 14, 8, 11), 2) == [(Side.SELL, 13.5)]


def test_stop_leg_gap_fills_at_open():
    broker = _broker()
    _long_bracket(broker)
    _fills(broker, _bar(10, 11.5, 10, 11), 1)
    assert _fills(broker, _bar(8, 8.5, 7, 8), 2) == [(Side.SELL, 8.0)]


def test_exit_legs_follow_fill_price():
    broker = _broker()
    _long_bracket(broker, exits_ref=11.0)
    assert _fills(broker, _bar(12, 12.5, 11.5, 12), 1) == [(Side.BUY, 12.0)]
    legs = {o.order_type: o for o in broker.open_orders()}
    assert legs[OrderType.STOP].stop_price == 10.0
    assert legs[OrderType.LIMIT].limit_price == 14.0


def test_entry_fill_carries_its_stop_loss():
    broker = _broker()
    _long_bracket(broker, exits_ref=11.0)
    (entry,) = broker.match(_bar(12, 12.5, 11.5, 12), bar_index=1)
    assert entry.stop_loss == 10.0
