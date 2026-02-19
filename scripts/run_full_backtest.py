#!/usr/bin/env python3
"""
Comprehensive Backtest Suite - Using IBKR Historical Data

Connects to IBKR Gateway/TWS and fetches precise historical data via the API,
then runs extensive backtests across multiple scenarios to validate the
Orchestrator algorithm.

Test Scenarios:
  1. Single-symbol tests (SPY, NVDA, AAPL)
  2. High-volatility test (SMCI)
  3. Multi-symbol portfolio test
  4. Default config vs Aggressive config
  5. Walk-forward analysis (H1 vs H2 2024)
  6. Extended period SPY (2024-2026)

Usage:
    python scripts/run_full_backtest.py
    python scripts/run_full_backtest.py --port 4002
    python scripts/run_full_backtest.py --symbols SPY NVDA --start 2024-06-01 --end 2024-12-31
"""

import argparse
import json
import os
import sys
import time
import math
import statistics
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_algo.backtest_v2 import (
    BacktestEngine,
    BacktestConfig,
    BacktestExporter,
    Bar,
    BacktestResults,
    DataProvider,
    DataRequest,
)
from trading_algo.orchestrator import (
    Orchestrator,
    create_orchestrator,
)
from trading_algo.orchestrator.config import create_aggressive_config
from trading_algo.config import IBKRConfig
from trading_algo.broker.ibkr import IBKRBroker

RESULTS_DIR = PROJECT_ROOT / "backtest_results"


def connect_ibkr(port: int = 4002) -> IBKRBroker:
    """Connect to IBKR Gateway/TWS."""
    config = IBKRConfig(host="127.0.0.1", port=port, client_id=55)
    broker = IBKRBroker(config=config, require_paper=False)
    broker.connect()
    return broker


def fetch_ibkr_data(
    broker: IBKRBroker,
    symbols: List[str],
    start_date: date,
    end_date: date,
    bar_size: str = "5 mins",
) -> Dict[str, List[Bar]]:
    """Fetch historical data from IBKR for all symbols."""
    provider = DataProvider(broker=broker)

    requests = [
        DataRequest(
            symbol=s,
            start_date=start_date,
            end_date=end_date,
            bar_size=bar_size,
        )
        for s in symbols
    ]

    def progress(pct, msg):
        bar_len = 30
        filled = int(pct * bar_len)
        bar_str = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar_str}] {pct*100:5.1f}% {msg[:60]:<60}", end="", flush=True)

    data = provider.get_data(requests, progress)
    print()  # newline after progress
    return data


def run_backtest(
    broker: IBKRBroker,
    name: str,
    trading_symbols: List[str],
    reference_symbols: List[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000,
    config_override: Optional[Any] = None,
    position_size_pct: float = 0.02,
    max_positions: int = 5,
) -> Optional[BacktestResults]:
    """Run a single backtest scenario with IBKR data."""
    print(f"\n{'='*70}")
    print(f"  BACKTEST: {name}")
    print(f"{'='*70}")
    print(f"  Symbols:   {', '.join(trading_symbols)}")
    print(f"  Reference: {', '.join(reference_symbols)}")
    print(f"  Period:    {start_date} to {end_date}")
    print(f"  Capital:   ${initial_capital:,.0f}")
    print(f"  Pos Size:  {position_size_pct*100:.1f}%")
    print()

    # Fetch data from IBKR
    all_symbols = list(set(trading_symbols + reference_symbols))
    print("  Fetching IBKR historical data...")
    try:
        data = fetch_ibkr_data(broker, all_symbols, start_date, end_date)
    except Exception as e:
        print(f"\n  ERROR fetching data: {e}")
        return None

    for symbol in all_symbols:
        if symbol in data and data[symbol]:
            bars = data[symbol]
            print(f"  {symbol}: {len(bars)} bars "
                  f"({bars[0].timestamp.date()} to {bars[-1].timestamp.date()})")
        else:
            print(f"  {symbol}: NO DATA")

    # Verify we have trading symbol data
    missing_trading = [s for s in trading_symbols if s not in data or not data[s]]
    if missing_trading:
        print(f"\n  ERROR: Missing data for trading symbols: {', '.join(missing_trading)}")
        print("  Skipping this test.\n")
        return None

    # Create strategy
    orchestrator = create_orchestrator(config_override)

    # Create engine config
    bt_config = BacktestConfig(
        strategy_name=name,
        symbols=trading_symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        max_positions=max_positions,
        bar_size="5 mins",
        data_source="IBKR",
    )

    # Run backtest
    engine = BacktestEngine(bt_config)
    t0 = time.time()

    def progress(pct, msg):
        bar_len = 30
        filled = int(pct * bar_len)
        bar_str = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar_str}] {pct*100:5.1f}% {msg[:50]:<50}", end="", flush=True)

    results = engine.run(orchestrator, data, progress)
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s ({results.bars_processed} bars processed)")

    return results


def print_results(results: BacktestResults, verbose: bool = True):
    """Print comprehensive results for a backtest."""
    m = results.metrics

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    pnl_color = GREEN if m.net_profit >= 0 else RED

    print(f"\n  {BOLD}--- Performance Summary ---{RESET}")
    print(f"  Total P&L:           {pnl_color}${m.net_profit:>12,.2f}{RESET}")
    print(f"  Total Return:        {pnl_color}{m.total_return_pct:>12.2f}%{RESET}")
    print(f"  Annualized Return:   {pnl_color}{m.annualized_return:>12.2f}%{RESET}")
    print(f"  Sharpe Ratio:        {m.sharpe_ratio:>12.2f}")
    print(f"  Sortino Ratio:       {m.sortino_ratio:>12.2f}")
    print(f"  Calmar Ratio:        {m.calmar_ratio:>12.2f}")
    print(f"  Max Drawdown:        {RED}{m.max_drawdown_pct:>12.2f}%{RESET}")

    print(f"\n  {BOLD}--- Trade Statistics ---{RESET}")
    print(f"  Total Trades:        {m.total_trades:>12}")
    print(f"  Winning Trades:      {m.winning_trades:>12}")
    print(f"  Losing Trades:       {m.losing_trades:>12}")

    wr_color = GREEN if m.win_rate >= 50 else YELLOW if m.win_rate >= 40 else RED
    print(f"  Win Rate:            {wr_color}{m.win_rate:>12.1f}%{RESET}")

    pf_color = GREEN if m.profit_factor >= 1.0 else RED
    print(f"  Profit Factor:       {pf_color}{m.profit_factor:>12.2f}{RESET}")
    print(f"  Expectancy:          ${m.expectancy:>11.2f}")
    print(f"  Avg Win:             ${m.avg_win:>11.2f}")
    print(f"  Avg Loss:            ${m.avg_loss:>11.2f}")
    print(f"  Largest Win:         ${m.largest_win:>11.2f}")
    print(f"  Largest Loss:        ${m.largest_loss:>11.2f}")
    print(f"  Avg Win/Loss Ratio:  {m.avg_win_loss_ratio:>12.2f}")

    print(f"\n  {BOLD}--- Daily Stats ---{RESET}")
    print(f"  Trading Days:        {m.trading_days:>12}")
    print(f"  Avg Daily P&L:       ${m.avg_daily_pnl:>11.2f}")
    print(f"  Daily P&L StdDev:    ${m.std_daily_pnl:>11.2f}")
    print(f"  Best Day:            ${m.best_day:>11.2f}")
    print(f"  Worst Day:           ${m.worst_day:>11.2f}")
    print(f"  Profitable Days:     {m.profitable_days:>12}")
    print(f"  Losing Days:         {m.losing_days:>12}")
    print(f"  Daily Win Rate:      {m.daily_win_rate:>12.1f}%")
    print(f"  Time in Market:      {m.time_in_market_pct:>12.1f}%")

    print(f"\n  {BOLD}--- Streaks ---{RESET}")
    print(f"  Max Win Streak:      {m.max_consecutive_wins:>12}")
    print(f"  Max Loss Streak:     {m.max_consecutive_losses:>12}")

    if verbose and results.trades:
        # Monthly breakdown
        monthly = results.get_monthly_returns()
        if monthly:
            print(f"\n  {BOLD}--- Monthly Returns ---{RESET}")
            for month, pnl in sorted(monthly.items()):
                mc = GREEN if pnl >= 0 else RED
                print(f"    {month}: {mc}${pnl:>10,.2f}{RESET}")

        # Exit reason breakdown
        exit_reasons: Dict[str, int] = {}
        for t in results.trades:
            reason = t.exit_reason
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        print(f"\n  {BOLD}--- Exit Reasons ---{RESET}")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25} {count:>5} ({count/len(results.trades)*100:.1f}%)")

        # Edge vote analysis on winning vs losing trades
        trades_with_edges = [t for t in results.trades if t.edge_votes]
        if trades_with_edges:
            print(f"\n  {BOLD}--- Edge Agreement Analysis ---{RESET}")
            winners = [t for t in trades_with_edges if t.is_winner]
            losers = [t for t in trades_with_edges if not t.is_winner]

            all_edges = set()
            for t in trades_with_edges:
                all_edges.update(t.edge_votes.keys())

            for edge_name in sorted(all_edges):
                win_long = sum(1 for t in winners if t.edge_votes.get(edge_name) == "LONG")
                win_short = sum(1 for t in winners if t.edge_votes.get(edge_name) == "SHORT")
                win_neutral = sum(1 for t in winners if t.edge_votes.get(edge_name) == "NEUTRAL")
                lose_long = sum(1 for t in losers if t.edge_votes.get(edge_name) == "LONG")
                lose_short = sum(1 for t in losers if t.edge_votes.get(edge_name) == "SHORT")
                lose_neutral = sum(1 for t in losers if t.edge_votes.get(edge_name) == "NEUTRAL")

                total_w = len(winners) or 1
                total_l = len(losers) or 1
                print(f"    {edge_name}:")
                print(f"      Winners: L={win_long}({win_long/total_w*100:.0f}%) "
                      f"S={win_short}({win_short/total_w*100:.0f}%) "
                      f"N={win_neutral}({win_neutral/total_w*100:.0f}%)")
                print(f"      Losers:  L={lose_long}({lose_long/total_l*100:.0f}%) "
                      f"S={lose_short}({lose_short/total_l*100:.0f}%) "
                      f"N={lose_neutral}({lose_neutral/total_l*100:.0f}%)")

        # Hourly P&L
        hourly = results.get_hourly_analysis()
        non_empty_hours = {h: v for h, v in hourly.items() if v["trades"] > 0}
        if non_empty_hours:
            print(f"\n  {BOLD}--- Hourly Performance ---{RESET}")
            for hour in sorted(non_empty_hours.keys()):
                v = non_empty_hours[hour]
                wr = (v["wins"] / v["trades"] * 100) if v["trades"] > 0 else 0
                hc = GREEN if v["pnl"] >= 0 else RED
                print(f"    {hour:02d}:00  {v['trades']:>4} trades  "
                      f"WR: {wr:>5.1f}%  P&L: {hc}${v['pnl']:>10,.2f}{RESET}")

        # Day-of-week
        weekday = results.get_weekday_analysis()
        non_empty_days = {d: v for d, v in weekday.items() if v["trades"] > 0}
        if non_empty_days:
            print(f"\n  {BOLD}--- Day-of-Week Performance ---{RESET}")
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                if day in non_empty_days:
                    v = non_empty_days[day]
                    wr = (v["wins"] / v["trades"] * 100) if v["trades"] > 0 else 0
                    dc = GREEN if v["pnl"] >= 0 else RED
                    print(f"    {day:<12} {v['trades']:>4} trades  "
                          f"WR: {wr:>5.1f}%  P&L: {dc}${v['pnl']:>10,.2f}{RESET}")


def run_all_backtests(broker: IBKRBroker):
    """Run the full backtest suite."""
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"

    print(f"\n{BOLD}{'#'*70}")
    print(f"#  COMPREHENSIVE ORCHESTRATOR BACKTEST SUITE")
    print(f"#  Using IBKR Precise Historical Data")
    print(f"{'#'*70}{RESET}\n")

    # Reference assets needed by Orchestrator for regime detection
    reference = ["SPY", "QQQ", "IWM"]
    all_results: Dict[str, BacktestResults] = {}

    # =========================================================================
    # TEST 1: Single-symbol tests (2024)
    # =========================================================================
    print(f"\n{BOLD}{'='*70}")
    print(f"  SECTION 1: SINGLE-SYMBOL BACKTESTS (2024)")
    print(f"{'='*70}{RESET}")

    for symbol in ["SPY", "NVDA", "AAPL"]:
        results = run_backtest(
            broker=broker,
            name=f"Single_{symbol}_2024",
            trading_symbols=[symbol],
            reference_symbols=reference,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        if results:
            all_results[f"single_{symbol}"] = results
            print_results(results, verbose=True)
        time.sleep(1)  # Pace IBKR requests

    # =========================================================================
    # TEST 2: High-volatility stock (SMCI)
    # =========================================================================
    print(f"\n{BOLD}{'='*70}")
    print(f"  SECTION 2: HIGH-VOLATILITY TEST (SMCI 2024)")
    print(f"{'='*70}{RESET}")

    results = run_backtest(
        broker=broker,
        name="HighVol_SMCI_2024",
        trading_symbols=["SMCI"],
        reference_symbols=reference,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )
    if results:
        all_results["smci_2024"] = results
        print_results(results, verbose=True)
    time.sleep(1)

    # =========================================================================
    # TEST 3: Multi-symbol portfolio
    # =========================================================================
    print(f"\n{BOLD}{'='*70}")
    print(f"  SECTION 3: MULTI-SYMBOL PORTFOLIO (2024)")
    print(f"{'='*70}{RESET}")

    results = run_backtest(
        broker=broker,
        name="Portfolio_2024",
        trading_symbols=["SPY", "NVDA", "AAPL", "GOOGL"],
        reference_symbols=reference,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        max_positions=4,
    )
    if results:
        all_results["portfolio_2024"] = results
        print_results(results, verbose=True)
    time.sleep(1)

    # =========================================================================
    # TEST 4: Aggressive config comparison
    # =========================================================================
    print(f"\n{BOLD}{'='*70}")
    print(f"  SECTION 4: AGGRESSIVE CONFIG vs DEFAULT")
    print(f"{'='*70}{RESET}")

    agg_config = create_aggressive_config(leverage=1.5)
    for symbol in ["SPY", "NVDA"]:
        results = run_backtest(
            broker=broker,
            name=f"Aggressive_{symbol}_2024",
            trading_symbols=[symbol],
            reference_symbols=reference,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            config_override=agg_config,
            position_size_pct=0.05,
            max_positions=5,
        )
        if results:
            all_results[f"aggressive_{symbol}"] = results
            print_results(results, verbose=False)
        time.sleep(1)

    # =========================================================================
    # TEST 5: Walk-forward analysis
    # =========================================================================
    print(f"\n{BOLD}{'='*70}")
    print(f"  SECTION 5: WALK-FORWARD ANALYSIS (SPY)")
    print(f"{'='*70}{RESET}")

    h1_results = run_backtest(
        broker=broker,
        name="WalkForward_SPY_H1_2024",
        trading_symbols=["SPY"],
        reference_symbols=reference,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )
    if h1_results:
        all_results["wf_spy_h1"] = h1_results
        print_results(h1_results, verbose=False)
    time.sleep(1)

    h2_results = run_backtest(
        broker=broker,
        name="WalkForward_SPY_H2_2024",
        trading_symbols=["SPY"],
        reference_symbols=reference,
        start_date=date(2024, 7, 1),
        end_date=date(2024, 12, 31),
    )
    if h2_results:
        all_results["wf_spy_h2"] = h2_results
        print_results(h2_results, verbose=False)
    time.sleep(1)

    # =========================================================================
    # TEST 6: Extended period SPY (2024-2026)
    # =========================================================================
    print(f"\n{BOLD}{'='*70}")
    print(f"  SECTION 6: EXTENDED PERIOD (SPY 2024-01 to 2026-01)")
    print(f"{'='*70}{RESET}")

    results = run_backtest(
        broker=broker,
        name="Extended_SPY_2024_2026",
        trading_symbols=["SPY"],
        reference_symbols=reference,
        start_date=date(2024, 1, 1),
        end_date=date(2026, 1, 31),
    )
    if results:
        all_results["extended_spy"] = results
        print_results(results, verbose=True)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n\n{BOLD}{'#'*70}")
    print(f"#  COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'#'*70}{RESET}\n")

    print(f"  {'Test':<30} {'Return':>9} {'Sharpe':>8} {'Win%':>7} "
          f"{'PF':>7} {'MaxDD':>8} {'Trades':>7} {'Expect':>9}")
    print(f"  {'-'*29} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*9}")

    for test_name, results in all_results.items():
        m = results.metrics
        ret_color = GREEN if m.total_return_pct >= 0 else RED
        print(
            f"  {test_name:<30} "
            f"{ret_color}{m.total_return_pct:>8.2f}%{RESET} "
            f"{m.sharpe_ratio:>8.2f} "
            f"{m.win_rate:>6.1f}% "
            f"{m.profit_factor:>7.2f} "
            f"{RED}{m.max_drawdown_pct:>7.2f}%{RESET} "
            f"{m.total_trades:>7} "
            f"${m.expectancy:>8.2f}"
        )

    # Overall assessment
    profitable = sum(1 for r in all_results.values() if r.metrics.net_profit > 0)
    total = len(all_results)

    print(f"\n  {BOLD}Overall: {profitable}/{total} tests profitable{RESET}")

    if total > 0:
        avg_return = statistics.mean([r.metrics.total_return_pct for r in all_results.values()])
        avg_sharpe = statistics.mean([r.metrics.sharpe_ratio for r in all_results.values()])
        avg_wr = statistics.mean([r.metrics.win_rate for r in all_results.values()])
        avg_pf = statistics.mean([r.metrics.profit_factor for r in all_results.values()])
        max_dd = max([r.metrics.max_drawdown_pct for r in all_results.values()])

        ret_c = GREEN if avg_return >= 0 else RED
        print(f"  Avg Return:       {ret_c}{avg_return:>8.2f}%{RESET}")
        print(f"  Avg Sharpe:       {avg_sharpe:>8.2f}")
        print(f"  Avg Win Rate:     {avg_wr:>8.1f}%")
        print(f"  Avg Profit Factor:{avg_pf:>8.2f}")
        print(f"  Worst Drawdown:   {RED}{max_dd:>8.2f}%{RESET}")

    # Export all results
    print(f"\n{BOLD}Exporting results...{RESET}")
    exporter = BacktestExporter()
    for test_name, results in all_results.items():
        try:
            export_path = exporter.export(results)
            print(f"  {test_name} -> {export_path}")
        except Exception as e:
            print(f"  {test_name}: Export error: {e}")

    # Save consolidated summary
    summary_path = RESULTS_DIR / f"full_backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_data = {}
    for test_name, results in all_results.items():
        summary_data[test_name] = {
            "config": results.config.to_dict(),
            "metrics": results.metrics.to_dict(),
            "trade_count": len(results.trades),
            "monthly_returns": results.get_monthly_returns(),
        }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"\n  Consolidated summary: {summary_path}")

    print(f"\n{BOLD}{'#'*70}")
    print(f"#  BACKTEST SUITE COMPLETE")
    print(f"{'#'*70}{RESET}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Orchestrator Backtest Suite (IBKR Data)")
    parser.add_argument("--port", type=int, default=4002, help="IBKR API port (default: 4002)")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to test (overrides full suite)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (for custom mode)")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (for custom mode)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital (default: 100000)")
    args = parser.parse_args()

    print("Connecting to IBKR Gateway...")
    try:
        broker = connect_ibkr(port=args.port)
        print(f"Connected on port {args.port}!")
    except Exception as e:
        print(f"ERROR: Could not connect to IBKR: {e}")
        print("Make sure TWS or IB Gateway is running with API enabled.")
        sys.exit(1)

    try:
        if args.symbols:
            # Custom single test
            start_date = date.fromisoformat(args.start) if args.start else date(2024, 1, 1)
            end_date = date.fromisoformat(args.end) if args.end else date(2024, 12, 31)
            reference = ["SPY", "QQQ", "IWM"]

            results = run_backtest(
                broker=broker,
                name=f"Custom_{'_'.join(args.symbols)}",
                trading_symbols=[s.upper() for s in args.symbols],
                reference_symbols=reference,
                start_date=start_date,
                end_date=end_date,
                initial_capital=args.capital,
            )
            if results:
                print_results(results, verbose=True)

                exporter = BacktestExporter()
                export_path = exporter.export(results)
                print(f"\n  Exported to: {export_path}")
        else:
            # Full suite
            run_all_backtests(broker)
    finally:
        print("\nDisconnecting from IBKR...")
        try:
            broker.disconnect()
            print("Disconnected.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
