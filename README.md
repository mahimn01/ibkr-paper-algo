# trading-algo

Multi-strategy quant trading system. US stocks through Interactive Brokers, crypto through Binance/Kraken/Hyperliquid. Runs strategies in parallel, has its own backtester, goes live through IBKR with safety gates.

## What's inside

| Piece | Where | What it does |
|---|---|---|
| Equity strategies | `trading_algo/multi_strategy/` | ~12 strategies plugged into one controller (momentum, mean reversion, pairs, ORB, flow, regime, Hurst, and more) |
| Crypto edges | `crypto_alpha/edges/` | 9 structural edges on perp futures (funding, basis, cross-exchange, intermarket cascades) |
| Options strategies | `trading_algo/quant_core/strategies/options/` | wheel, PMCC, enhanced wheel, jade lizard, portfolio wheel, hybrid regime, put spreads |
| ATLAS | `trading_algo/quant_core/models/atlas/` | Deep-RL trader, Mamba + cross-attention, PPO + EWC (v7) |
| IBKR data/ops CLI | `trading_algo/ibkr_tool.py` | ~46 commands for quotes, chains, depth, history, scanners, orders, what-if calcs |
| Flex CLI | `trading_algo/flex_tool.py` | ~31 commands for Flex Web Service (statements, P&L, cash, dividends) |
| Gemini trader | `trading_algo/llm/` | Chat and trader loop. Stopped driving trades through it because literature doesn't back LLM trading signals |
| RAT | `trading_algo/rat/` | Experimental research (reflexivity, attention topology, adversarial algo detection) |

## Backtester

The whole system is worthless if the backtests lie, so it got a lot of attention.

- Signals on bar N fill at bar N+1 open (no same-bar look-ahead)
- VWAP tracking when adding to existing positions
- Backward-only data lookups
- Realistic commissions (~$0.0035/share) and slippage (2bps)
- Walk-forward validation across sequential folds
- Overfitting checked with PBO, deflated Sharpe, and White's reality check

The crypto runner adds funding settlement at the correct UTC hours, leverage tracking, liquidation simulation with a 0.5% penalty, and 365-day annualization. There's also a fraud detection suite that runs null tests. Random signals should give Sharpe near zero, reversed signals should flip the PnL, doubled costs should crush the edges, and single-asset isolation shows where alpha actually comes from. If any fail, the infrastructure is lying.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

| Command | What it does |
|---|---|
| `python scripts/run_10yr_backtest.py` | 10-year equity backtest |
| `python crypto_alpha/scripts/deep_analysis.py` | Crypto deep analysis |
| `python crypto_alpha/scripts/fraud_detection.py` | Crypto fraud tests |
| `python scripts/validate_edge.py --symbols NQ,ES --fetch` | Edge validation (ORB, gap fade, VWAP) |
| `python run.py --paper` | Paper trading via IB Gateway (port 4002) |
| `python run.py --live` | Live trading, after clearing safety gates |

Results land in `backtest_results/`.

## Live trading safety

Every gate has to clear before an order transmits.

| Gate | Requirement |
|---|---|
| `TRADING_ALLOW_LIVE` | Must be `true` in env |
| `--allow-live` | Must be on the command line |
| Paper-only enforcement | Must be explicitly disabled |
| Per-order confirmation | Type `YES` at the terminal for every place/modify/cancel/bracket |
| Confirmation callback | Must be wired into the CLI, otherwise orders get blocked outright |

Most of my live trading runs through IB Gateway via IBC in tmux, with `AutoRestartTime=23:55` so it survives IBKR's nightly disconnect. Paper and live can run on different ports at the same time.

## Latest results (V11 config)

| System | Sharpe | Return | Max DD | Period |
|---|---|---|---|---|
| Equity V11 | 0.48 | +151.3% | 18.5% | 2016–2026 |
| Crypto 9-edge | 0.28 | +7.7% | 21.8% | 2022–2026 |

Equity has DSR 16 with p ≈ 0, OOS Sharpe 0.55, alpha vs SPY around 4% annualized, beta 0.12.

Crypto lags BTC buy-and-hold (0.80 Sharpe). Only three edges are positive as standalones. IMC (+0.72), CED (+0.40), PBMR (+0.33).

## Known issues

- Crypto system lags BTC buy-and-hold (it's a diversification play, not a replacement)
- `_strategy_positions` in the controller never gets populated, so per-strategy position limits are dormant
- V11 has trailing stops off (they hurt Sharpe on every config I tested)
- Crypto funding data has gaps depending on which exchange API it pulls from
- ATLAS v7 training needs the R3000 dataset, not shipped in the repo
- Edge validation wants `pandas>=2.0.0` and `statsmodels>=0.14.0`

## Docs

`docs/` has more detail.

- `ARCHITECTURE.md`, deep dive
- `SAFETY.md`, full live-trading safety model
- `LLM_TRADER.md`, Gemini loop and chat
- `WORKFLOWS.md`, day-to-day workflows
- `DB_SCHEMA.md`, sqlite audit schema
- `TRADITIONAL_VS_AI_TRADING_VERDICT.md`, why the LLM direction got shelved
- `HOW_LLMS_WERE_USED_IN_RESEARCH.md`, critique of LLM alpha discovery papers
- `CLAUDE.md` at the repo root, non-negotiable trading rules

`CHANGELOG.md` has the full commit history.
