"""
Shared backtest metrics computation.

Used by both the equity runner (252 trading days) and the crypto runner
(365 trading days) to avoid duplicating ~150 lines of identical statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from trading_algo.multi_strategy.backtest_runner import (
    MultiStrategyBacktestResults,
    StrategyAttribution,
)


def compute_backtest_metrics(
    equity_curve: List[float],
    daily_returns: List[float],
    timestamps: List[Any],
    trades: List[Dict],
    signals_by_strategy: Dict[str, int],
    closed_trades: int,
    winning_trades: int,
    initial_capital: float,
    risk_free_rate: float = 0.045,
    trading_days_per_year: int = 252,
    benchmark_daily_returns: Optional[np.ndarray] = None,
) -> MultiStrategyBacktestResults:
    """
    Compute all backtest metrics from raw data.

    Args:
        equity_curve: List of equity values (including initial capital).
        daily_returns: List of daily percentage returns.
        timestamps: List of timestamps corresponding to daily returns.
        trades: List of trade dicts from the runner.
        signals_by_strategy: Dict of strategy_name -> signal count.
        closed_trades: Total number of closed round-trip trades.
        winning_trades: Number of profitable closed trades.
        initial_capital: Starting capital.
        risk_free_rate: Annual risk-free rate (default 4.5%).
        trading_days_per_year: 252 for equities, 365 for crypto.
        benchmark_daily_returns: Optional benchmark returns for alpha/beta.

    Returns:
        MultiStrategyBacktestResults with all metrics filled.
    """
    tdy = trading_days_per_year
    ec = np.array(equity_curve) if equity_curve else np.array([initial_capital])
    dr = np.array(daily_returns) if daily_returns else np.array([0.0])

    # ── Core returns ──
    total_return = (ec[-1] / ec[0]) - 1 if ec[0] > 0 else 0
    n_years = max(len(dr) / tdy, 1 / tdy)
    ann_return = (1 + total_return) ** (1 / n_years) - 1

    # ── Volatility ──
    vol = float(np.std(dr, ddof=1) * np.sqrt(tdy)) if len(dr) > 1 else 0.15

    # ── Sharpe ratio ──
    daily_rf = (1 + risk_free_rate) ** (1 / tdy) - 1
    excess = dr - daily_rf
    if len(dr) > 1 and np.std(excess, ddof=1) > 1e-10:
        sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(tdy))
    else:
        sharpe = 0.0

    # ── Sortino ratio ──
    downside_diff = np.minimum(dr - daily_rf, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside_diff ** 2)))
    sortino = (
        float(np.mean(excess) * np.sqrt(tdy) / downside_dev)
        if downside_dev > 1e-10 else 0.0
    )

    # ── Drawdown ──
    peak = np.maximum.accumulate(ec)
    dd = (peak - ec) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd))

    dd_duration = 0
    max_dd_duration = 0
    for d in dd:
        if d > 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    # ── Win rate ──
    total_trade_count = closed_trades
    win_rate = winning_trades / closed_trades if closed_trades > 0 else 0.0

    # ── Calmar ──
    calmar = ann_return / max_dd if max_dd > 1e-10 else 0.0

    # ── VaR / CVaR (95%) ──
    if len(dr) >= 10:
        var_95 = float(-np.percentile(dr, 5))
        tail = dr[dr <= np.percentile(dr, 5)]
        cvar_95 = float(-np.mean(tail)) if len(tail) > 0 else var_95
    else:
        var_95 = cvar_95 = 0.0

    # ── Skewness / Kurtosis ──
    if len(dr) > 2:
        mean_dr = np.mean(dr)
        std_dr = np.std(dr, ddof=1)
        if std_dr > 1e-10:
            skewness = float(np.mean(((dr - mean_dr) / std_dr) ** 3))
            kurtosis = float(np.mean(((dr - mean_dr) / std_dr) ** 4) - 3)
        else:
            skewness = kurtosis = 0.0
    else:
        skewness = kurtosis = 0.0

    # ── Profit factor / Expectancy ──
    trade_pnls = _extract_trade_pnls(trades)
    if len(trade_pnls) > 0:
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        total_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
        total_losses = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 1e-10 else 0.0
        expectancy = float(np.mean(trade_pnls))
    else:
        profit_factor = expectancy = 0.0

    # ── Annual turnover ──
    shares_key = "shares" if trades and "shares" in trades[0] else "size"
    total_traded_value = sum(
        abs(t.get(shares_key, 0) * t["price"]) for t in trades
    )
    avg_equity = float(np.mean(ec)) if len(ec) > 0 else initial_capital
    annual_turnover = (
        total_traded_value / avg_equity / n_years
        if avg_equity > 0 and n_years > 0 else 0.0
    )

    # ── Benchmark-relative metrics ──
    beta = alpha_annual = information_ratio = benchmark_corr = 0.0

    if benchmark_daily_returns is not None and len(benchmark_daily_returns) >= 10:
        bench = benchmark_daily_returns
        min_len = min(len(dr), len(bench))
        algo_r = dr[:min_len]
        bench_r = bench[:min_len]

        cov_matrix = np.cov(algo_r, bench_r)
        var_bench = cov_matrix[1, 1]
        if var_bench > 1e-10:
            beta = float(cov_matrix[0, 1] / var_bench)

        bench_ann = float(np.mean(bench_r) * tdy)
        algo_ann = float(np.mean(algo_r) * tdy)
        alpha_annual = algo_ann - risk_free_rate - beta * (bench_ann - risk_free_rate)

        active_returns = algo_r - bench_r
        tracking_error = float(np.std(active_returns, ddof=1) * np.sqrt(tdy))
        if tracking_error > 1e-10:
            information_ratio = float(np.mean(active_returns) * tdy / tracking_error)

        corr = np.corrcoef(algo_r, bench_r)
        benchmark_corr = float(corr[0, 1]) if not np.isnan(corr[0, 1]) else 0.0

    # ── Strategy attribution ──
    attribution = {
        name: StrategyAttribution(name=name, n_signals=count)
        for name, count in signals_by_strategy.items()
    }

    return MultiStrategyBacktestResults(
        total_return=total_return,
        annualized_return=ann_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        volatility=vol,
        total_trades=total_trade_count,
        win_rate=win_rate,
        calmar_ratio=calmar,
        profit_factor=profit_factor,
        expectancy_per_trade=expectancy,
        skewness=skewness,
        kurtosis=kurtosis,
        var_95=var_95,
        cvar_95=cvar_95,
        max_drawdown_duration_days=max_dd_duration,
        annual_turnover=annual_turnover,
        beta=beta,
        alpha_annual=alpha_annual,
        information_ratio=information_ratio,
        benchmark_correlation=benchmark_corr,
        equity_curve=ec.tolist(),
        daily_returns=dr.tolist(),
        timestamps=timestamps,
        strategy_attribution=attribution,
    )


def _extract_trade_pnls(trades: List[Dict]) -> np.ndarray:
    """Extract per-trade P&L from the trade log.

    Works with both equity trades (shares/BUY/SELL/COVER) and
    crypto trades (size/CLOSE_LONG/CLOSE_SHORT with explicit pnl).
    """
    # Crypto trades have explicit PnL on close entries
    explicit_pnls = [
        t["pnl"] for t in trades
        if "pnl" in t and t.get("side", "").startswith("CLOSE")
    ]
    if explicit_pnls:
        return np.array(explicit_pnls)

    # Equity trades: pair open/close by symbol
    open_trades: Dict[str, List[Dict]] = {}
    pnls: List[float] = []

    for t in trades:
        sym = t["symbol"]
        if t["side"] in ("BUY", "SHORT"):
            open_trades.setdefault(sym, []).append(t)
        elif t["side"] in ("SELL", "COVER"):
            opens = open_trades.get(sym, [])
            if opens:
                entry = opens.pop(0)
                entry_price = entry["price"]
                exit_price = t["price"]
                shares = abs(entry.get("shares", entry.get("size", 0)))
                if entry["side"] == "BUY":
                    pnl = (exit_price - entry_price) * shares
                else:
                    pnl = (entry_price - exit_price) * shares
                pnls.append(pnl)

    return np.array(pnls) if pnls else np.array([])
