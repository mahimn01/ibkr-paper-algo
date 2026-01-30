"""
IBKR Stock Scanner - Live Pre-Market Stock Selection.

Connects the stock selector system to IBKR for real data.
Scans a universe of stocks, analyzes volatility/volume/momentum/technicals,
and returns the best day trading candidates.

Two scanning modes:
1. Full Scan: Pull 60 days of daily bars per stock, run all analyzers.
   Best for pre-market prep. Takes ~5-10 minutes due to IBKR pacing.
2. Today's Movers: Quick scan using latest bar to find today's biggest
   movers (gaps, intraday surges). Takes ~2-3 minutes.

This scanner uses IBKR historical data for the analyzers that work with
pure price data: Volatility, Volume, Momentum, Technical.
News/Options/Social analyzers use neutral defaults since they need
external API connections.

IBKR Pacing: IBKR rate-limits historical data requests. This scanner
adds delays between requests to avoid pacing violations.

Usage:
    from trading_algo.stock_selector.ibkr_scanner import IBKRStockScanner

    scanner = IBKRStockScanner(broker)

    # Full pre-market scan
    candidates = scanner.scan(top_n=5)

    # Quick today's movers scan
    movers = scanner.scan_todays_movers(top_n=10)

    # Mid-day rescan (reuses cached data where possible)
    refreshed = scanner.scan_todays_movers(top_n=5)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.instruments import InstrumentSpec
from trading_algo.stock_selector.analyzers import (
    PriceBar,
    VolatilityAnalyzer,
    VolumeAnalyzer,
    MomentumAnalyzer,
    TechnicalAnalyzer,
)
from trading_algo.stock_selector.models import (
    MarketRegime,
    StockCandidate,
    SelectionResult,
    NewsScore,
    SocialScore,
    OptionsScore,
    ShortInterestScore,
    RiskMetrics,
    VolatilityScore,
)
from trading_algo.stock_selector.scoring import (
    CompositeScorer,
    rank_candidates,
)


# Expanded universe of liquid US stocks suitable for day trading
# Focused on high-volume, volatile names
DAY_TRADE_UNIVERSE = [
    # Mega-cap tech (very liquid, often volatile)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # High-vol tech
    "AMD", "INTC", "CRM", "NFLX", "PYPL", "SQ", "SHOP", "SNOW",
    "UBER", "ABNB", "DKNG", "RBLX", "SNAP", "PINS",
    # Semiconductors (volatile, high volume)
    "MU", "MRVL", "QCOM", "AVGO", "TSM", "SMCI", "ARM", "MARA",
    # Financials
    "JPM", "BAC", "GS", "MS", "C", "WFC", "SCHW",
    # Energy (volatile with oil)
    "XOM", "CVX", "OXY", "SLB", "DVN", "MPC",
    # EV / Clean energy (high beta)
    "RIVN", "LCID", "NIO", "PLUG", "ENPH", "FSLR",
    # Popular / meme stocks (high retail volume, volatile)
    "GME", "AMC", "PLTR", "SOFI", "HOOD", "COIN", "MSTR",
    # Chinese ADRs (gap-heavy, volatile)
    "BABA", "JD", "PDD", "BIDU", "LI",
    # Biotech (catalyst-driven, volatile)
    "MRNA", "BNTX", "REGN",
    # Airlines / Travel (volatile, macro-sensitive)
    "UAL", "DAL", "AAL", "CCL", "RCL",
    # Consumer
    "DIS", "NKE", "SBUX", "MCD", "WMT", "TGT", "COST",
    # Healthcare
    "UNH", "ABBV", "PFE", "JNJ",
    # ETFs (for reference / trading)
    "SPY", "QQQ", "IWM", "SOXL", "TQQQ",
]


@dataclass
class ScanResult:
    """Result of an IBKR stock scan."""
    timestamp: datetime
    market_regime: MarketRegime
    candidates: List[StockCandidate]
    total_scanned: int
    symbols_with_data: int
    passed_filters: int
    scan_time_seconds: float
    errors: List[str] = field(default_factory=list)


class IBKRStockScanner:
    """
    Scans stocks via IBKR and identifies the best day trading candidates.

    Uses real IBKR historical data for analysis. Focuses on:
    - Volatility (ATR%, intraday range, gaps)
    - Volume (liquidity, relative volume, volume momentum)
    - Momentum (multi-timeframe returns, RSI, trend strength)
    - Technical (support/resistance, breakout patterns)

    The scanner pulls daily bars for each symbol, runs the analyzers,
    scores candidates, and returns ranked results.
    """

    def __init__(
        self,
        broker: IBKRBroker,
        universe: Optional[List[str]] = None,
        exchange: str = "SMART",
        currency: str = "USD",
        min_price: float = 5.0,
        max_price: float = 5000.0,
        min_atr_pct: float = 0.015,  # 1.5% minimum daily ATR for day trading
        min_avg_volume: float = 500_000,
        data_days: int = 60,  # Days of historical data to pull
        ibkr_pacing_delay: float = 1.0,  # Seconds between IBKR requests (avoid pacing violations)
    ):
        self.broker = broker
        self.universe = universe or DAY_TRADE_UNIVERSE.copy()
        self.exchange = exchange
        self.currency = currency
        self.min_price = min_price
        self.max_price = max_price
        self.min_atr_pct = min_atr_pct
        self.min_avg_volume = min_avg_volume
        self.data_days = data_days
        self.pacing_delay = ibkr_pacing_delay

        # Initialize analyzers
        self.volatility_analyzer = VolatilityAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()

        # Data cache (persists across scans for mid-day rescan efficiency)
        self._price_cache: Dict[str, List[PriceBar]] = {}
        self._last_request_time: float = 0.0

    def scan(
        self,
        top_n: int = 5,
        market_regime: Optional[MarketRegime] = None,
        custom_universe: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> ScanResult:
        """
        Run the complete stock scanning and selection process.

        Args:
            top_n: Number of top candidates to return
            market_regime: Override detected market regime
            custom_universe: Override default universe
            verbose: Print progress

        Returns:
            ScanResult with top candidates
        """
        start_time = time.time()
        universe = custom_universe or self.universe
        errors = []

        if verbose:
            print(f"\nScanning {len(universe)} stocks...")

        # Clear cache
        self._price_cache.clear()

        # Pull SPY data first (for market regime and relative strength)
        spy_bars = self._pull_bars("SPY", verbose=False)

        # Detect market regime
        if market_regime is None:
            market_regime = self._detect_regime(spy_bars)
            if verbose:
                print(f"  Market Regime: {market_regime.name}")

        # Initialize scorer
        scorer = CompositeScorer(regime=market_regime)

        # Scan each symbol
        candidates = []
        symbols_with_data = 0

        for i, symbol in enumerate(universe):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Scanned {i + 1}/{len(universe)}...")

            try:
                # Pull data
                bars = self._pull_bars(symbol, verbose=False)
                if bars is None or len(bars) < 30:
                    continue

                symbols_with_data += 1

                # Quick filters before expensive analysis
                current_price = bars[-1].close
                if current_price < self.min_price or current_price > self.max_price:
                    continue

                # Run analysis
                candidate = self._analyze(symbol, bars, spy_bars)
                if candidate is None:
                    continue

                # Apply quality filters
                if not self._passes_filters(candidate):
                    continue

                # Score
                candidate = scorer.score(candidate)
                candidates.append(candidate)

            except Exception as e:
                errors.append(f"{symbol}: {e}")
                continue

        # Rank and return top N
        top_candidates = rank_candidates(candidates, top_n)

        scan_time = time.time() - start_time

        if verbose:
            print(f"\n  Scanned: {len(universe)} | Data: {symbols_with_data} | "
                  f"Passed: {len(candidates)} | Time: {scan_time:.1f}s")
            if errors:
                print(f"  Errors: {len(errors)}")

        return ScanResult(
            timestamp=datetime.now(),
            market_regime=market_regime,
            candidates=top_candidates,
            total_scanned=len(universe),
            symbols_with_data=symbols_with_data,
            passed_filters=len(candidates),
            scan_time_seconds=scan_time,
            errors=errors,
        )

    def scan_todays_movers(
        self,
        top_n: int = 10,
        min_gap_pct: float = 0.02,  # 2% gap minimum
        verbose: bool = True,
    ) -> ScanResult:
        """
        Quick scan for today's biggest movers (gaps + intraday surges).

        This is much faster than a full scan because it only needs 2 days
        of data per stock to detect today's gap and move.

        Use this for:
        - Pre-market scanning (find stocks gapping big)
        - Mid-day rescan (find stocks surging today)

        Args:
            top_n: Number of top movers to return
            min_gap_pct: Minimum gap/move to consider (default 2%)
            verbose: Print progress

        Returns:
            ScanResult with today's top movers
        """
        start_time = time.time()
        errors = []

        if verbose:
            print(f"\nScanning for today's movers ({len(self.universe)} stocks)...")

        movers: List[Dict] = []

        for i, symbol in enumerate(self.universe):
            if verbose and (i + 1) % 20 == 0:
                print(f"  Checked {i + 1}/{len(self.universe)}...")

            try:
                self._respect_pacing()

                instrument = InstrumentSpec(
                    kind="STK",
                    symbol=symbol,
                    exchange=self.exchange,
                    currency=self.currency,
                )

                # Pull just 2 days of data (yesterday close + today)
                bars = self.broker.get_historical_bars(
                    instrument,
                    duration="2 D",
                    bar_size="1 day",
                    what_to_show="TRADES",
                    use_rth=True,
                )

                if not bars or len(bars) < 2:
                    continue

                prev_close = bars[-2].close
                today_open = bars[-1].open
                today_close = bars[-1].close
                today_high = bars[-1].high
                today_low = bars[-1].low
                today_volume = bars[-1].volume or 0

                if prev_close <= 0:
                    continue

                # Gap: open vs yesterday's close
                gap_pct = (today_open - prev_close) / prev_close
                # Intraday move: current vs open
                intraday_pct = (today_close - today_open) / today_open if today_open > 0 else 0
                # Total move: current vs yesterday close
                total_move_pct = (today_close - prev_close) / prev_close
                # Intraday range
                range_pct = (today_high - today_low) / prev_close if prev_close > 0 else 0

                # Filter: need significant movement
                if abs(total_move_pct) < min_gap_pct and range_pct < min_gap_pct * 1.5:
                    continue

                # Score by total magnitude of movement (direction doesn't matter for selection)
                move_score = abs(total_move_pct) * 50 + range_pct * 30 + abs(gap_pct) * 20

                movers.append({
                    'symbol': symbol,
                    'price': today_close,
                    'gap_pct': gap_pct,
                    'intraday_pct': intraday_pct,
                    'total_move_pct': total_move_pct,
                    'range_pct': range_pct,
                    'volume': today_volume,
                    'move_score': move_score,
                })

            except Exception as e:
                errors.append(f"{symbol}: {e}")
                continue

        # Sort by move_score descending
        movers.sort(key=lambda x: x['move_score'], reverse=True)
        top_movers = movers[:top_n]

        scan_time = time.time() - start_time

        if verbose:
            print(f"\n  Found {len(movers)} movers | Time: {scan_time:.1f}s")
            if top_movers:
                print(f"\n  {'Rank':<5} {'Symbol':<8} {'Price':>8} {'Gap':>7} {'Move':>7} {'Range':>7} {'Score':>7}")
                print("  " + "-" * 55)
                for i, m in enumerate(top_movers, 1):
                    print(f"  {i:<5} {m['symbol']:<8} ${m['price']:>7.2f} "
                          f"{m['gap_pct']*100:>+6.1f}% {m['total_move_pct']*100:>+6.1f}% "
                          f"{m['range_pct']*100:>6.1f}% {m['move_score']:>7.1f}")

        # Now do full analysis on the top movers
        if verbose and top_movers:
            print(f"\n  Running full analysis on top {len(top_movers)} movers...")

        # Pull full data for the top movers
        spy_bars = self._pull_bars("SPY", verbose=False)
        regime = self._detect_regime(spy_bars)
        scorer = CompositeScorer(regime=regime)

        candidates = []
        for m in top_movers:
            symbol = m['symbol']
            try:
                bars = self._pull_bars(symbol, verbose=False)
                if bars is None or len(bars) < 30:
                    continue

                candidate = self._analyze(symbol, bars, spy_bars)
                if candidate is None:
                    continue

                # Boost score for today's movers
                candidate = scorer.score(candidate)
                # Add bonus for today's movement magnitude
                mover_bonus = min(15, m['move_score'] * 0.5)
                candidate.composite_score = min(100, candidate.composite_score + mover_bonus)

                candidates.append(candidate)
            except Exception as e:
                errors.append(f"{symbol} (analysis): {e}")

        top_candidates = rank_candidates(candidates, top_n)

        return ScanResult(
            timestamp=datetime.now(),
            market_regime=regime,
            candidates=top_candidates,
            total_scanned=len(self.universe),
            symbols_with_data=len(movers),
            passed_filters=len(candidates),
            scan_time_seconds=time.time() - start_time,
            errors=errors,
        )

    def _respect_pacing(self):
        """Wait if needed to respect IBKR pacing limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.pacing_delay:
            time.sleep(self.pacing_delay - elapsed)
        self._last_request_time = time.time()

    def _pull_bars(
        self,
        symbol: str,
        verbose: bool = False,
    ) -> Optional[List[PriceBar]]:
        """Pull daily bars from IBKR for a symbol (with pacing)."""
        if symbol in self._price_cache:
            return self._price_cache[symbol]

        try:
            # Respect IBKR pacing limits
            self._respect_pacing()

            instrument = InstrumentSpec(
                kind="STK",
                symbol=symbol,
                exchange=self.exchange,
                currency=self.currency,
            )

            bars = self.broker.get_historical_bars(
                instrument,
                duration=f"{self.data_days} D",
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True,
            )

            if not bars:
                return None

            price_bars = [
                PriceBar(
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume or 0,
                )
                for b in bars
            ]

            self._price_cache[symbol] = price_bars

            if verbose:
                print(f"  {symbol}: {len(price_bars)} bars")

            return price_bars

        except Exception as e:
            if verbose:
                print(f"  {symbol}: Error - {e}")
            return None

    def _analyze(
        self,
        symbol: str,
        bars: List[PriceBar],
        spy_bars: Optional[List[PriceBar]],
    ) -> Optional[StockCandidate]:
        """Run multi-factor analysis on a stock."""

        # Volatility analysis
        volatility = self.volatility_analyzer.analyze(bars)
        if volatility is None:
            return None

        # Volume analysis
        volume = self.volume_analyzer.analyze(bars)
        if volume is None:
            return None

        # Quick filter: minimum liquidity
        if volume.liquidity_score < 30:
            return None

        # Momentum analysis
        momentum = self.momentum_analyzer.analyze(bars, spy_bars)
        if momentum is None:
            return None

        # Technical analysis
        technical = self.technical_analyzer.analyze(bars)
        if technical is None:
            return None

        # Create candidate with neutral scores for non-price analyzers
        return StockCandidate(
            symbol=symbol,
            company_name=symbol,
            sector=_get_sector(symbol),
            industry="",
            market_cap=0.0,
            price=bars[-1].close,
            timestamp=datetime.now(),
            volatility=volatility,
            volume=volume,
            momentum=momentum,
            technical=technical,
            news=_neutral_news_score(),
            social=_neutral_social_score(),
            options=_neutral_options_score(),
            short_interest=_neutral_short_score(),
            risk=self._assess_risk(bars, volume),
            catalysts=[],
        )

    def _passes_filters(self, candidate: StockCandidate) -> bool:
        """Check if candidate passes day trading quality filters."""
        # Minimum volatility (must have enough range to profit)
        if candidate.volatility.atr_percent < self.min_atr_pct:
            return False

        # Minimum liquidity
        if candidate.volume.liquidity_score < 40:
            return False

        # Not too volatile (hard to manage risk)
        if candidate.volatility.atr_percent > 0.10:  # >10% daily ATR
            return False

        # Must be tradeable
        if not candidate.risk.tradeable:
            return False

        # Minimum average volume
        if candidate.volume.avg_volume_20d < self.min_avg_volume:
            return False

        return True

    def _assess_risk(
        self,
        bars: List[PriceBar],
        volume,
    ) -> RiskMetrics:
        """Assess trading risk."""
        # Bid-ask spread estimate
        avg_range = sum((b.high - b.low) / b.close for b in bars[-20:]) / 20
        estimated_spread = avg_range * 0.1

        # Gap risk
        gaps = []
        for i in range(1, len(bars)):
            gap = abs(bars[i].open - bars[i - 1].close) / bars[i - 1].close
            gaps.append(gap)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        gap_risk = min(1.0, avg_gap * 20)

        # Risk score
        risk_score = 30
        if estimated_spread > 0.005:
            risk_score += 20
        if gap_risk > 0.3:
            risk_score += 15
        if volume.liquidity_score < 50:
            risk_score += 15

        tradeable = (
            volume.liquidity_score >= 30
            and estimated_spread < 0.02
            and risk_score < 80
        )

        return RiskMetrics(
            earnings_days_away=None,
            has_binary_event=False,
            halt_risk=0.01,
            bid_ask_spread_pct=estimated_spread,
            slippage_estimate=estimated_spread * 0.5,
            gap_risk=gap_risk,
            tail_risk=gap_risk * 0.5,
            risk_score=risk_score,
            tradeable=tradeable,
        )

    def _detect_regime(
        self,
        spy_bars: Optional[List[PriceBar]],
    ) -> MarketRegime:
        """Detect market regime from SPY data."""
        if spy_bars is None or len(spy_bars) < 30:
            return MarketRegime.NEUTRAL

        current_price = spy_bars[-1].close
        sma_20 = sum(b.close for b in spy_bars[-20:]) / 20
        price_vs_sma = (current_price - sma_20) / sma_20

        # Volatility
        returns = [
            (spy_bars[i].close - spy_bars[i - 1].close) / spy_bars[i - 1].close
            for i in range(-20, 0)
        ]
        volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5 * (252**0.5)

        if volatility > 0.35:
            return MarketRegime.CRISIS
        elif volatility > 0.25:
            return MarketRegime.BEAR if price_vs_sma < -0.03 else MarketRegime.NEUTRAL
        elif price_vs_sma > 0.06:
            return MarketRegime.STRONG_BULL
        elif price_vs_sma > 0.03:
            return MarketRegime.BULL
        elif price_vs_sma < -0.03:
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL


def print_scan_results(result: ScanResult):
    """Pretty print scan results."""
    print("\n" + "=" * 80)
    print("DAY TRADING STOCK SCANNER RESULTS")
    print("=" * 80)
    print(f"Timestamp:    {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Regime:       {result.market_regime.name}")
    print(f"Scanned:      {result.total_scanned} stocks")
    print(f"Had Data:     {result.symbols_with_data}")
    print(f"Passed:       {result.passed_filters}")
    print(f"Scan Time:    {result.scan_time_seconds:.1f}s")
    print()

    if not result.candidates:
        print("  No candidates found!")
        return

    print(f"{'Rank':<5} {'Symbol':<8} {'Score':>6} {'Price':>8} {'ATR%':>6} "
          f"{'Vol20d':>10} {'Mom':>5} {'Dir':<6} {'Top Factors'}")
    print("-" * 80)

    for c in result.candidates:
        direction = "LONG" if c.recommended_direction > 0 else (
            "SHORT" if c.recommended_direction < 0 else "EITHER"
        )
        vol_str = f"{c.volume.avg_volume_20d / 1e6:.1f}M"
        factors = ", ".join(c.top_factors[:2]) if c.top_factors else ""
        print(f"{c.rank:<5} {c.symbol:<8} {c.composite_score:>6.1f} "
              f"${c.price:>7.2f} {c.volatility.atr_percent * 100:>5.2f}% "
              f"{vol_str:>10} {c.momentum.momentum_score:>5.0f} "
              f"{direction:<6} {factors[:30]}")

    print("=" * 80)

    # Detailed top 3
    print("\nDETAILED TOP 3:")
    for c in result.candidates[:3]:
        print(f"\n  {c.symbol} - Score: {c.composite_score:.1f}")
        print(f"  Price: ${c.price:.2f}")
        print(f"  ATR%: {c.volatility.atr_percent * 100:.2f}%")
        print(f"  Avg Volume: {c.volume.avg_volume_20d / 1e6:.1f}M")
        print(f"  Momentum: {c.momentum.momentum_score:.0f}")
        print(f"  RSI: {c.momentum.rsi_14:.0f}")
        print(f"  Trend Strength: {c.momentum.trend_strength:.0f}")
        print(f"  Direction: {'LONG' if c.recommended_direction > 0 else 'SHORT' if c.recommended_direction < 0 else 'EITHER'}")
        print(f"  Expected Move: {c.expected_move * 100:.1f}%")
        if c.top_factors:
            print(f"  Factors: {', '.join(c.top_factors[:3])}")


def _get_sector(symbol: str) -> str:
    """Simple sector lookup."""
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "AMZN": "Consumer", "NVDA": "Technology", "META": "Technology",
        "TSLA": "Automotive", "AMD": "Technology", "INTC": "Technology",
        "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
        "XOM": "Energy", "CVX": "Energy", "OXY": "Energy",
        "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
        "DIS": "Consumer", "NKE": "Consumer", "WMT": "Consumer",
        "COIN": "Crypto", "MSTR": "Crypto", "MARA": "Crypto",
    }
    return sector_map.get(symbol, "Unknown")


def _neutral_news_score() -> NewsScore:
    """Return neutral news score."""
    return NewsScore(
        news_count_24h=0, news_count_7d=0, news_velocity=0,
        news_recency_hours=168, sentiment_score=0, sentiment_magnitude=0,
        sentiment_trend=0, mainstream_sentiment=0, financial_sentiment=0,
        social_sentiment=0, has_breaking_news=False, has_analyst_action=False,
        has_insider_activity=False, news_score=30, catalyst_score=30,
    )


def _neutral_social_score() -> SocialScore:
    """Return neutral social score."""
    return SocialScore(
        reddit_mentions_24h=0, reddit_mentions_trend=0, reddit_sentiment=0,
        twitter_mentions_24h=0, twitter_sentiment=0, influencer_mentions=0,
        stocktwits_sentiment=0, stocktwits_volume=0, social_momentum=0,
        viral_potential=0, retail_interest_score=30,
    )


def _neutral_options_score() -> OptionsScore:
    """Return neutral options score."""
    return OptionsScore(
        options_volume=0, put_call_ratio=1.0, unusual_activity=False,
        call_flow_score=50, put_flow_score=50, net_premium=0,
        iv_rank=50, iv_skew=0, large_trades_detected=0,
        sweep_orders_detected=0, smart_money_direction=0,
        options_signal_score=30, squeeze_potential=30,
    )


def _neutral_short_score() -> ShortInterestScore:
    """Return neutral short interest score."""
    return ShortInterestScore(
        short_interest_ratio=0.05, days_to_cover=2.0,
        short_interest_change=0, cost_to_borrow=1.0,
        shares_available=1_000_000, utilization=0.3,
        squeeze_setup_score=30, short_pressure_score=30,
    )
