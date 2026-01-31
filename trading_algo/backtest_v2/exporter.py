"""
Backtest results exporter.

Exports all backtest data to various formats for deep analysis:
- JSON: Complete data export
- CSV: Trades, daily P&L, signals
- Charts: Equity curve, drawdowns, monthly returns
- HTML: Interactive dashboard report
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import BacktestResults, BacktestTrade, DailyResult, DrawdownPeriod


class BacktestExporter:
    """
    Exports backtest results to multiple formats.

    Creates a comprehensive export folder with:
    - summary.json: Complete results in JSON
    - trades.csv: All trades with details
    - daily_pnl.csv: Daily profit/loss breakdown
    - equity_curve.csv: Equity over time
    - metrics.json: Key performance metrics
    - config.json: Backtest configuration
    - report.html: Interactive HTML report
    - charts/: PNG charts (if matplotlib available)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("backtest_results")
        self._has_matplotlib = self._check_matplotlib()

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available for charting."""
        try:
            import matplotlib
            return True
        except ImportError:
            return False

    def export(
        self,
        results: BacktestResults,
        folder_name: Optional[str] = None,
    ) -> Path:
        """
        Export all backtest results to a folder.

        Args:
            results: BacktestResults object
            folder_name: Optional custom folder name

        Returns:
            Path to the export folder
        """
        # Create export folder
        if folder_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbols = "_".join(results.config.symbols[:3])
            if len(results.config.symbols) > 3:
                symbols += f"_plus{len(results.config.symbols) - 3}"
            folder_name = f"backtest_{symbols}_{timestamp}"

        export_path = self.output_dir / folder_name
        export_path.mkdir(parents=True, exist_ok=True)

        # Export all components
        self._export_summary_json(results, export_path)
        self._export_trades_csv(results, export_path)
        self._export_daily_pnl_csv(results, export_path)
        self._export_equity_curve_csv(results, export_path)
        self._export_metrics_json(results, export_path)
        self._export_config_json(results, export_path)
        self._export_drawdowns_csv(results, export_path)
        self._export_monthly_returns_csv(results, export_path)

        if self._has_matplotlib:
            self._export_charts(results, export_path)

        self._export_html_report(results, export_path)

        return export_path

    def _export_summary_json(self, results: BacktestResults, path: Path) -> None:
        """Export complete results to JSON."""
        data = {
            "config": {
                "symbols": results.config.symbols,
                "start_date": results.config.start_date.isoformat(),
                "end_date": results.config.end_date.isoformat(),
                "initial_capital": results.config.initial_capital,
                "commission_per_share": results.config.commission_per_share,
                "slippage_pct": results.config.slippage_pct,
                "max_position_size": results.config.max_position_size,
                "bar_size": results.config.bar_size,
            },
            "metrics": self._metrics_to_dict(results.metrics),
            "trades_count": len(results.trades),
            "daily_results_count": len(results.daily_results),
            "equity_points_count": len(results.equity_curve),
            "drawdown_periods_count": len(results.drawdown_periods),
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(path / "summary.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _export_trades_csv(self, results: BacktestResults, path: Path) -> None:
        """Export all trades to CSV."""
        if not results.trades:
            return

        fieldnames = [
            "trade_id", "symbol", "direction", "entry_time", "exit_time",
            "entry_price", "exit_price", "quantity", "pnl", "pnl_pct",
            "commission", "slippage", "hold_time_minutes", "mae", "mfe",
            "mae_pct", "mfe_pct", "exit_reason", "signal_strength",
        ]

        with open(path / "trades.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trade in results.trades:
                hold_time = 0
                if trade.exit_time and trade.entry_time:
                    hold_time = (trade.exit_time - trade.entry_time).total_seconds() / 60

                row = {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "entry_time": trade.entry_time.isoformat() if trade.entry_time else "",
                    "exit_time": trade.exit_time.isoformat() if trade.exit_time else "",
                    "entry_price": round(trade.entry_price, 4),
                    "exit_price": round(trade.exit_price, 4) if trade.exit_price else "",
                    "quantity": trade.quantity,
                    "pnl": round(trade.pnl, 2) if trade.pnl else 0,
                    "pnl_pct": round(trade.pnl_pct * 100, 4) if trade.pnl_pct else 0,
                    "commission": round(trade.commission, 2),
                    "slippage": round(trade.slippage, 2),
                    "hold_time_minutes": round(hold_time, 1),
                    "mae": round(trade.mae, 2) if trade.mae else 0,
                    "mfe": round(trade.mfe, 2) if trade.mfe else 0,
                    "mae_pct": round(trade.mae_pct * 100, 4) if trade.mae_pct else 0,
                    "mfe_pct": round(trade.mfe_pct * 100, 4) if trade.mfe_pct else 0,
                    "exit_reason": trade.exit_reason or "",
                    "signal_strength": round(trade.signal_strength, 2) if trade.signal_strength else "",
                }
                writer.writerow(row)

    def _export_daily_pnl_csv(self, results: BacktestResults, path: Path) -> None:
        """Export daily P&L breakdown to CSV."""
        if not results.daily_results:
            return

        fieldnames = [
            "date", "pnl", "pnl_pct", "trades", "winners", "losers",
            "win_rate", "gross_profit", "gross_loss", "largest_win",
            "largest_loss", "avg_win", "avg_loss", "profit_factor",
            "starting_equity", "ending_equity", "max_drawdown",
            "max_drawdown_pct",
        ]

        with open(path / "daily_pnl.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for day in results.daily_results:
                total_trades = day.winners + day.losers
                win_rate = day.winners / total_trades if total_trades > 0 else 0
                avg_win = day.gross_profit / day.winners if day.winners > 0 else 0
                avg_loss = day.gross_loss / day.losers if day.losers > 0 else 0
                profit_factor = abs(day.gross_profit / day.gross_loss) if day.gross_loss != 0 else float('inf')

                row = {
                    "date": day.date.isoformat(),
                    "pnl": round(day.pnl, 2),
                    "pnl_pct": round(day.pnl_pct * 100, 4),
                    "trades": total_trades,
                    "winners": day.winners,
                    "losers": day.losers,
                    "win_rate": round(win_rate * 100, 2),
                    "gross_profit": round(day.gross_profit, 2),
                    "gross_loss": round(day.gross_loss, 2),
                    "largest_win": round(day.largest_win, 2),
                    "largest_loss": round(day.largest_loss, 2),
                    "avg_win": round(avg_win, 2),
                    "avg_loss": round(avg_loss, 2),
                    "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
                    "starting_equity": round(day.starting_equity, 2),
                    "ending_equity": round(day.ending_equity, 2),
                    "max_drawdown": round(day.max_drawdown, 2),
                    "max_drawdown_pct": round(day.max_drawdown_pct * 100, 4),
                }
                writer.writerow(row)

    def _export_equity_curve_csv(self, results: BacktestResults, path: Path) -> None:
        """Export equity curve to CSV."""
        if not results.equity_curve:
            return

        fieldnames = ["timestamp", "equity", "drawdown", "drawdown_pct"]

        with open(path / "equity_curve.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            peak = results.config.initial_capital
            for point in results.equity_curve:
                if point.equity > peak:
                    peak = point.equity
                drawdown = peak - point.equity
                drawdown_pct = drawdown / peak if peak > 0 else 0

                row = {
                    "timestamp": point.timestamp.isoformat(),
                    "equity": round(point.equity, 2),
                    "drawdown": round(drawdown, 2),
                    "drawdown_pct": round(drawdown_pct * 100, 4),
                }
                writer.writerow(row)

    def _export_metrics_json(self, results: BacktestResults, path: Path) -> None:
        """Export key metrics to JSON."""
        data = self._metrics_to_dict(results.metrics)

        with open(path / "metrics.json", "w") as f:
            json.dump(data, f, indent=2)

    def _export_config_json(self, results: BacktestResults, path: Path) -> None:
        """Export backtest configuration to JSON."""
        data = {
            "symbols": results.config.symbols,
            "start_date": results.config.start_date.isoformat(),
            "end_date": results.config.end_date.isoformat(),
            "initial_capital": results.config.initial_capital,
            "commission_per_share": results.config.commission_per_share,
            "slippage_pct": results.config.slippage_pct,
            "max_position_size": results.config.max_position_size,
            "bar_size": results.config.bar_size,
            "algorithm_name": results.config.algorithm_name,
            "algorithm_params": results.config.algorithm_params,
        }

        with open(path / "config.json", "w") as f:
            json.dump(data, f, indent=2)

    def _export_drawdowns_csv(self, results: BacktestResults, path: Path) -> None:
        """Export drawdown periods to CSV."""
        if not results.drawdown_periods:
            return

        fieldnames = [
            "start_date", "end_date", "recovery_date", "duration_days",
            "recovery_days", "max_drawdown", "max_drawdown_pct",
            "peak_equity", "trough_equity",
        ]

        with open(path / "drawdowns.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for dd in results.drawdown_periods:
                row = {
                    "start_date": dd.start_date.isoformat(),
                    "end_date": dd.end_date.isoformat() if dd.end_date else "",
                    "recovery_date": "",  # Not tracked in current model
                    "duration_days": dd.duration_days,
                    "recovery_days": dd.recovery_days if dd.recovery_days else 0,
                    "max_drawdown": round(dd.drawdown_amount, 2),
                    "max_drawdown_pct": round(dd.drawdown_percent, 4),
                    "peak_equity": round(dd.peak_equity, 2),
                    "trough_equity": round(dd.trough_equity, 2),
                }
                writer.writerow(row)

    def _export_monthly_returns_csv(self, results: BacktestResults, path: Path) -> None:
        """Export monthly returns to CSV."""
        if not results.daily_results:
            return

        # Aggregate daily results into monthly
        monthly: Dict[str, float] = {}
        for day in results.daily_results:
            key = day.date.strftime("%Y-%m")
            monthly[key] = monthly.get(key, 0) + day.net_pnl

        fieldnames = ["month", "pnl", "pnl_pct"]

        with open(path / "monthly_returns.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            initial = results.config.initial_capital
            for month, pnl in sorted(monthly.items()):
                row = {
                    "month": month,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / initial * 100, 4),
                }
                writer.writerow(row)

    def _export_charts(self, results: BacktestResults, path: Path) -> None:
        """Export charts as PNG images."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            return

        charts_dir = path / "charts"
        charts_dir.mkdir(exist_ok=True)

        # Equity curve chart
        self._create_equity_chart(results, charts_dir, plt, mdates)

        # Drawdown chart
        self._create_drawdown_chart(results, charts_dir, plt, mdates)

        # Daily P&L distribution
        self._create_daily_pnl_chart(results, charts_dir, plt)

        # Monthly returns heatmap
        self._create_monthly_returns_chart(results, charts_dir, plt)

        # Win/Loss distribution
        self._create_trade_distribution_chart(results, charts_dir, plt)

        plt.close('all')

    def _create_equity_chart(self, results: BacktestResults, path: Path, plt: Any, mdates: Any) -> None:
        """Create equity curve chart."""
        if not results.equity_curve:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        timestamps = [p.timestamp for p in results.equity_curve]
        equity = [p.equity for p in results.equity_curve]

        ax.plot(timestamps, equity, 'b-', linewidth=1)
        ax.axhline(y=results.config.initial_capital, color='gray', linestyle='--', alpha=0.5)

        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)

        # Fill profit/loss areas
        ax.fill_between(timestamps, results.config.initial_capital, equity,
                       where=[e >= results.config.initial_capital for e in equity],
                       alpha=0.3, color='green', label='Profit')
        ax.fill_between(timestamps, results.config.initial_capital, equity,
                       where=[e < results.config.initial_capital for e in equity],
                       alpha=0.3, color='red', label='Loss')

        ax.legend()
        plt.tight_layout()
        fig.savefig(path / "equity_curve.png", dpi=150)
        plt.close(fig)

    def _create_drawdown_chart(self, results: BacktestResults, path: Path, plt: Any, mdates: Any) -> None:
        """Create drawdown chart."""
        if not results.equity_curve:
            return

        fig, ax = plt.subplots(figsize=(12, 4))

        timestamps = []
        drawdowns = []
        peak = results.config.initial_capital

        for point in results.equity_curve:
            if point.equity > peak:
                peak = point.equity
            dd_pct = (peak - point.equity) / peak * 100
            timestamps.append(point.timestamp)
            drawdowns.append(-dd_pct)

        ax.fill_between(timestamps, 0, drawdowns, color='red', alpha=0.5)
        ax.plot(timestamps, drawdowns, 'r-', linewidth=0.5)

        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(path / "drawdown.png", dpi=150)
        plt.close(fig)

    def _create_daily_pnl_chart(self, results: BacktestResults, path: Path, plt: Any) -> None:
        """Create daily P&L distribution chart."""
        if not results.daily_results:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        pnls = [d.net_pnl for d in results.daily_results]
        dates = [d.date for d in results.daily_results]

        # Bar chart of daily P&L
        colors = ['green' if p >= 0 else 'red' for p in pnls]
        ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_title('Daily P&L', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trading Day')
        ax1.set_ylabel('P&L ($)')
        ax1.grid(True, alpha=0.3, axis='y')

        # Histogram of daily P&L
        ax2.hist(pnls, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
        mean_pnl = sum(pnls) / len(pnls) if pnls else 0
        ax2.axvline(x=mean_pnl, color='green', linestyle='--', linewidth=1, label=f'Mean: ${mean_pnl:.2f}')
        ax2.set_title('Daily P&L Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('P&L ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(path / "daily_pnl.png", dpi=150)
        plt.close(fig)

    def _create_monthly_returns_chart(self, results: BacktestResults, path: Path, plt: Any) -> None:
        """Create monthly returns bar chart."""
        if not results.daily_results:
            return

        # Aggregate monthly returns
        monthly: Dict[str, float] = {}
        for day in results.daily_results:
            key = day.date.strftime("%Y-%m")
            monthly[key] = monthly.get(key, 0) + day.net_pnl

        if not monthly:
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        months = sorted(monthly.keys())
        values = [monthly[m] for m in months]
        colors = ['green' if v >= 0 else 'red' for v in values]

        ax.bar(months, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linewidth=0.5)

        ax.set_title('Monthly Returns', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('P&L ($)')
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(path / "monthly_returns.png", dpi=150)
        plt.close(fig)

    def _create_trade_distribution_chart(self, results: BacktestResults, path: Path, plt: Any) -> None:
        """Create trade P&L distribution chart."""
        if not results.trades:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        pnls = [t.net_pnl for t in results.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Win/Loss pie chart
        if wins or losses:
            sizes = [len(wins), len(losses)]
            labels = [f'Winners ({len(wins)})', f'Losers ({len(losses)})']
            colors_pie = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0)
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax1.set_title('Win/Loss Ratio', fontsize=14, fontweight='bold')

        # Trade P&L histogram
        if pnls:
            ax2.hist(pnls, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
            avg_pnl = sum(pnls) / len(pnls)
            ax2.axvline(x=avg_pnl, color='green', linestyle='--', linewidth=1,
                       label=f'Avg: ${avg_pnl:.2f}')
            ax2.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('P&L ($)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(path / "trade_distribution.png", dpi=150)
        plt.close(fig)

    def _export_html_report(self, results: BacktestResults, path: Path) -> None:
        """Export interactive HTML report."""
        metrics = results.metrics
        config = results.config

        # Calculate additional stats
        total_trades = len(results.trades)
        winning_trades = len([t for t in results.trades if t.net_pnl > 0])
        losing_trades = len([t for t in results.trades if t.net_pnl <= 0])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {config.algorithm_name or 'Strategy'}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            color: #00d4ff;
            font-size: 2.5em;
        }}
        h2 {{
            color: #00d4ff;
            margin: 20px 0 15px;
            border-bottom: 2px solid #00d4ff;
            padding-bottom: 10px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .card h3 {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #2a3a5e;
        }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #888; }}
        .metric-value {{ font-weight: bold; }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #f39c12; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #2a3a5e;
        }}
        th {{
            background: #0f3460;
            color: #00d4ff;
            font-weight: 600;
        }}
        tr:hover {{ background: #1a3a5e; }}
        .chart-container {{ margin: 20px 0; text-align: center; }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .summary-bar {{
            display: flex;
            justify-content: space-around;
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        .summary-item {{ text-align: center; }}
        .summary-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .summary-label {{
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        .config-item {{
            background: #0f3460;
            padding: 10px 15px;
            border-radius: 6px;
        }}
        footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            border-top: 1px solid #2a3a5e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Report</h1>
        <p style="text-align: center; color: #888; margin-bottom: 30px;">
            {config.algorithm_name or 'Strategy'} |
            {config.start_date.isoformat()} to {config.end_date.isoformat()} |
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>

        <div class="summary-bar">
            <div class="summary-item">
                <div class="summary-value {'positive' if metrics.net_profit >= 0 else 'negative'}">
                    ${metrics.net_profit:,.2f}
                </div>
                <div class="summary-label">Total P&L</div>
            </div>
            <div class="summary-item">
                <div class="summary-value {'positive' if metrics.total_return_pct >= 0 else 'negative'}">
                    {metrics.total_return_pct:.2f}%
                </div>
                <div class="summary-label">Total Return</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{metrics.sharpe_ratio:.2f}</div>
                <div class="summary-label">Sharpe Ratio</div>
            </div>
            <div class="summary-item">
                <div class="summary-value neutral">{metrics.win_rate:.1f}%</div>
                <div class="summary-label">Win Rate</div>
            </div>
            <div class="summary-item">
                <div class="summary-value negative">-{metrics.max_drawdown_pct:.2f}%</div>
                <div class="summary-label">Max Drawdown</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>📈 Performance Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Total P&L</span>
                    <span class="metric-value {'positive' if metrics.net_profit >= 0 else 'negative'}">
                        ${metrics.net_profit:,.2f}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Return</span>
                    <span class="metric-value {'positive' if metrics.total_return_pct >= 0 else 'negative'}">
                        {metrics.total_return_pct:.2f}%
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Annualized Return</span>
                    <span class="metric-value {'positive' if metrics.annualized_return >= 0 else 'negative'}">
                        {metrics.annualized_return:.2f}%
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Daily P&L</span>
                    <span class="metric-value {'positive' if metrics.avg_daily_pnl >= 0 else 'negative'}">
                        ${metrics.avg_daily_pnl:,.2f}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Best Day</span>
                    <span class="metric-value positive">${metrics.best_day:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Worst Day</span>
                    <span class="metric-value negative">${metrics.worst_day:,.2f}</span>
                </div>
            </div>

            <div class="card">
                <h3>📊 Risk Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Sharpe Ratio</span>
                    <span class="metric-value">{metrics.sharpe_ratio:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sortino Ratio</span>
                    <span class="metric-value">{metrics.sortino_ratio:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Calmar Ratio</span>
                    <span class="metric-value">{metrics.calmar_ratio:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Drawdown</span>
                    <span class="metric-value negative">${metrics.max_drawdown:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Drawdown %</span>
                    <span class="metric-value negative">{metrics.max_drawdown_pct:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Volatility (Ann.)</span>
                    <span class="metric-value">{metrics.std_daily_pnl * 100:.2f}%</span>
                </div>
            </div>

            <div class="card">
                <h3>🎯 Trade Statistics</h3>
                <div class="metric">
                    <span class="metric-label">Total Trades</span>
                    <span class="metric-value">{metrics.total_trades}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Winning Trades</span>
                    <span class="metric-value positive">{metrics.winning_trades}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Losing Trades</span>
                    <span class="metric-value negative">{metrics.losing_trades}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Win Rate</span>
                    <span class="metric-value">{metrics.win_rate:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Profit Factor</span>
                    <span class="metric-value">{metrics.profit_factor:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Expectancy</span>
                    <span class="metric-value {'positive' if metrics.expectancy >= 0 else 'negative'}">
                        ${metrics.expectancy:,.2f}
                    </span>
                </div>
            </div>

            <div class="card">
                <h3>📉 Trade Analysis</h3>
                <div class="metric">
                    <span class="metric-label">Average Win</span>
                    <span class="metric-value positive">${metrics.avg_win:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Loss</span>
                    <span class="metric-value negative">${metrics.avg_loss:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Largest Win</span>
                    <span class="metric-value positive">${metrics.largest_win:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Largest Loss</span>
                    <span class="metric-value negative">${metrics.largest_loss:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Win Streak</span>
                    <span class="metric-value">{metrics.max_consecutive_wins}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Loss Streak</span>
                    <span class="metric-value">{metrics.max_consecutive_losses}</span>
                </div>
            </div>
        </div>

        {'<h2>📈 Charts</h2>' if self._has_matplotlib else ''}
        {'<div class="chart-container"><img src="charts/equity_curve.png" alt="Equity Curve"></div>' if self._has_matplotlib and (path / "charts/equity_curve.png").exists() else ''}
        {'<div class="chart-container"><img src="charts/drawdown.png" alt="Drawdown"></div>' if self._has_matplotlib and (path / "charts/drawdown.png").exists() else ''}
        {'<div class="chart-container"><img src="charts/daily_pnl.png" alt="Daily P&L"></div>' if self._has_matplotlib and (path / "charts/daily_pnl.png").exists() else ''}
        {'<div class="chart-container"><img src="charts/monthly_returns.png" alt="Monthly Returns"></div>' if self._has_matplotlib and (path / "charts/monthly_returns.png").exists() else ''}
        {'<div class="chart-container"><img src="charts/trade_distribution.png" alt="Trade Distribution"></div>' if self._has_matplotlib and (path / "charts/trade_distribution.png").exists() else ''}

        <h2>⚙️ Configuration</h2>
        <div class="card">
            <div class="config-grid">
                <div class="config-item">
                    <span class="metric-label">Symbols:</span><br>
                    <strong>{', '.join(config.symbols)}</strong>
                </div>
                <div class="config-item">
                    <span class="metric-label">Period:</span><br>
                    <strong>{config.start_date} to {config.end_date}</strong>
                </div>
                <div class="config-item">
                    <span class="metric-label">Initial Capital:</span><br>
                    <strong>${config.initial_capital:,.2f}</strong>
                </div>
                <div class="config-item">
                    <span class="metric-label">Commission:</span><br>
                    <strong>${config.commission_per_share}/share</strong>
                </div>
                <div class="config-item">
                    <span class="metric-label">Slippage:</span><br>
                    <strong>{config.slippage_pct * 100:.3f}%</strong>
                </div>
                <div class="config-item">
                    <span class="metric-label">Max Position:</span><br>
                    <strong>{config.max_position_size} shares</strong>
                </div>
                <div class="config-item">
                    <span class="metric-label">Bar Size:</span><br>
                    <strong>{config.bar_size}</strong>
                </div>
                <div class="config-item">
                    <span class="metric-label">Algorithm:</span><br>
                    <strong>{config.algorithm_name or 'Custom'}</strong>
                </div>
            </div>
        </div>

        <h2>📋 Recent Trades</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Qty</th>
                    <th>P&L</th>
                    <th>P&L %</th>
                </tr>
            </thead>
            <tbody>
                {''.join(self._format_trade_row(t) for t in results.trades[-20:])}
            </tbody>
        </table>
        <p style="color: #888; margin-top: 10px;">Showing last 20 trades. See trades.csv for complete list.</p>

        <h2>📅 Daily P&L Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>P&L</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Largest Win</th>
                    <th>Largest Loss</th>
                </tr>
            </thead>
            <tbody>
                {''.join(self._format_daily_row(d) for d in results.daily_results[-30:])}
            </tbody>
        </table>
        <p style="color: #888; margin-top: 10px;">Showing last 30 days. See daily_pnl.csv for complete list.</p>

        <footer>
            <p>Generated by Backtest Engine v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>"""

        with open(path / "report.html", "w") as f:
            f.write(html)

    def _format_trade_row(self, trade: BacktestTrade) -> str:
        """Format a trade as an HTML table row."""
        pnl_class = "positive" if trade.net_pnl > 0 else "negative"
        return f"""
            <tr>
                <td>{trade.symbol}</td>
                <td>{trade.direction}</td>
                <td>{trade.entry_time.strftime('%Y-%m-%d %H:%M') if trade.entry_time else ''}</td>
                <td>{trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else ''}</td>
                <td>${trade.entry_price:.2f}</td>
                <td>${trade.exit_price:.2f if trade.exit_price else 0:.2f}</td>
                <td>{trade.quantity}</td>
                <td class="{pnl_class}">${trade.net_pnl:.2f}</td>
                <td class="{pnl_class}">{trade.pnl_percent:.2f}%</td>
            </tr>"""

    def _format_daily_row(self, day: DailyResult) -> str:
        """Format a daily result as an HTML table row."""
        pnl_class = "positive" if day.net_pnl >= 0 else "negative"
        total_trades = day.trades_won + day.trades_lost
        win_rate = day.trades_won / total_trades * 100 if total_trades > 0 else 0
        return f"""
            <tr>
                <td>{day.date.isoformat()}</td>
                <td class="{pnl_class}">${day.net_pnl:.2f}</td>
                <td>{total_trades}</td>
                <td>{win_rate:.1f}%</td>
                <td class="positive">${day.gross_pnl:.2f}</td>
                <td class="negative">-</td>
            </tr>"""

    def _metrics_to_dict(self, metrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_pnl": round(metrics.net_profit, 2),
            "total_return": round(metrics.total_return, 2),
            "total_return_pct": round(metrics.total_return_pct, 4),
            "annualized_return": round(metrics.annualized_return, 4),
            "cagr": round(metrics.cagr, 4),
            "sharpe_ratio": round(metrics.sharpe_ratio, 2),
            "sortino_ratio": round(metrics.sortino_ratio, 2),
            "calmar_ratio": round(metrics.calmar_ratio, 2),
            "max_drawdown": round(metrics.max_drawdown, 2),
            "max_drawdown_pct": round(metrics.max_drawdown_pct, 4),
            "volatility": round(metrics.std_daily_pnl, 4),
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": round(metrics.win_rate, 4),
            "profit_factor": round(metrics.profit_factor, 2),
            "expectancy": round(metrics.expectancy, 2),
            "avg_win": round(metrics.avg_win, 2),
            "avg_loss": round(metrics.avg_loss, 2),
            "largest_win": round(metrics.largest_win, 2),
            "largest_loss": round(metrics.largest_loss, 2),
            "avg_trade": round(metrics.avg_trade, 2),
            "avg_daily_pnl": round(metrics.avg_daily_pnl, 2),
            "best_day": round(metrics.best_day, 2),
            "worst_day": round(metrics.worst_day, 2),
            "max_consecutive_wins": metrics.max_consecutive_wins,
            "max_consecutive_losses": metrics.max_consecutive_losses,
            "avg_hold_time_minutes": round(metrics.avg_trade_duration_minutes, 1),
            "time_in_market_pct": round(metrics.time_in_market_pct, 2),
        }
