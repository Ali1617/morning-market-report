"""
Per-stock optimizer — tune params independently for each ticker.

For every ticker, try:
    - both strategies (MA-cross, VWAP-pullback)
    - all 3 intraday timeframes (5m, 10m, 15m)
    - 2 search seeds

For each (ticker, TF, strategy, seed) combo, run a 250-iter guided random search
scoring against PER-STOCK targets (PnL ≥ 5%, Win ≥ 55%, PF ≥ 2.0, ≥5 trades).
Pick the single best combo for each ticker. Output a portfolio JSON.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from optimizer import (
    TICKERS, HERE, get_data,
    backtest_one as backtest_macross,
    PARAM_GRID as MA_GRID,
    rand_params as ma_rand,
    mutate as ma_mutate,
)
from optimizer_vwap import (
    backtest_vwap,
    VWAP_GRID,
    rand_params as vwap_rand,
    mutate as vwap_mutate,
)

# Per-stock targets — applied to single-stock metrics
TARGETS_PS = {"pnl_pct": 5.0, "win_rate": 0.55, "profit_factor": 2.0}
MIN_TRADES = 5
TIMEFRAMES = ["5m", "10m", "15m"]
SEEDS = [1, 42]
N_ITERS = 250


def hits_single(m: dict) -> bool:
    pf = m["profit_factor"]
    if pf == float("inf"):
        pf = 100.0
    return (
        m["pnl_pct"] >= TARGETS_PS["pnl_pct"]
        and m["win_rate"] >= TARGETS_PS["win_rate"]
        and pf >= TARGETS_PS["profit_factor"]
    )


def score_single(m: dict) -> float:
    if m["n_trades"] < MIN_TRADES:
        return -1e6 + m["n_trades"]
    pf = min(m["profit_factor"], 10.0) if m["profit_factor"] != float("inf") else 10.0
    pnl_term = m["pnl_pct"] / TARGETS_PS["pnl_pct"]
    win_term = m["win_rate"] / TARGETS_PS["win_rate"]
    pf_term = min(pf, 5.0) / TARGETS_PS["profit_factor"]
    base = pnl_term * win_term * pf_term
    return (1000.0 + base) if hits_single(m) else base


def optimize_ticker_strategy(df: pd.DataFrame, strategy_name: str,
                             n_iters: int = N_ITERS, seed: int = 42) -> dict | None:
    if strategy_name == "ma_cross":
        bt = backtest_macross
        rand_fn = ma_rand
        mutate_fn = ma_mutate
    elif strategy_name == "vwap":
        bt = backtest_vwap
        rand_fn = vwap_rand
        mutate_fn = vwap_mutate
    else:
        raise ValueError(strategy_name)

    rng = random.Random(seed)
    best = None
    best_s = -float("inf")

    for i in range(n_iters):
        if best is not None and rng.random() < 0.30:
            params = mutate_fn(best["params"], rng)
        else:
            params = rand_fn(rng)
        try:
            _, m = bt(df, params)
        except Exception:
            continue
        s = score_single(m)
        if s > best_s:
            best_s = s
            best = {"params": params, "metrics": m, "score": s, "strategy": strategy_name}

    return best


def run_per_stock() -> dict:
    portfolio: dict = {}

    # Cache all data first so we don't re-download
    data_cache = {tf: get_data(tf) for tf in TIMEFRAMES}

    for ticker in TICKERS:
        print(f"\n{'═' * 80}")
        print(f"OPTIMIZING {ticker}")
        print(f"{'═' * 80}")

        candidates = []
        for tf in TIMEFRAMES:
            df = data_cache[tf].get(ticker)
            if df is None or len(df) < 80:
                continue
            for strategy in ["ma_cross", "vwap"]:
                for seed in SEEDS:
                    res = optimize_ticker_strategy(df, strategy, n_iters=N_ITERS, seed=seed)
                    if res is None:
                        continue
                    res["tf"] = tf
                    res["ticker"] = ticker
                    res["seed"] = seed
                    candidates.append(res)
                    m = res["metrics"]
                    pf_disp = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "inf"
                    hits = "✓" if hits_single(m) else " "
                    print(
                        f"  {tf:>3} {strategy:<9} s={seed}  "
                        f"PnL={m['pnl_pct']:+6.1f}%  Win={m['win_rate']*100:4.0f}%  "
                        f"PF={pf_disp:>5}  N={m['n_trades']:>3}  score={res['score']:8.2f}  {hits}"
                    )

        if candidates:
            best = max(candidates, key=lambda x: x["score"])
            portfolio[ticker] = best
            m = best["metrics"]
            pf_disp = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "inf"
            print(
                f"\n  >>> WINNER for {ticker}: {best['tf']} / {best['strategy']}  "
                f"PnL {m['pnl_pct']:+.1f}%  Win {m['win_rate']*100:.0f}%  PF {pf_disp}  "
                f"N={m['n_trades']}  hits_target={hits_single(m)}"
            )

    return portfolio


def print_summary(portfolio: dict) -> None:
    print(f"\n{'═' * 100}")
    print("PORTFOLIO SUMMARY (per-stock tuning)")
    print(f"{'═' * 100}")
    print(
        f"  {'Ticker':>6}  {'TF':>4}  {'Strategy':<10}  "
        f"{'PnL':>7}  {'Win':>5}  {'PF':>6}  {'N':>3}  {'Hits':>5}"
    )
    print(f"  {'─' * 96}")
    n_hit = 0
    pnl_sum = 0.0
    for tk, w in portfolio.items():
        m = w["metrics"]
        hits = hits_single(m)
        if hits:
            n_hit += 1
        pnl_sum += m["pnl_pct"]
        pf_disp = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "  inf"
        print(
            f"  {tk:>6}  {w['tf']:>4}  {w['strategy']:<10}  "
            f"{m['pnl_pct']:+7.1f}  {m['win_rate']*100:4.0f}%  {pf_disp:>6}  "
            f"{m['n_trades']:>3}  {'✓ YES' if hits else ' no':>5}"
        )
    avg_pnl = pnl_sum / len(portfolio) if portfolio else 0.0
    print(f"\n  Aggregate avg PnL across 8 stocks: {avg_pnl:+.1f}%")
    print(f"  Stocks hitting all 3 targets:      {n_hit}/{len(portfolio)}")


def main():
    t = time.time()
    portfolio = run_per_stock()
    print_summary(portfolio)

    out = HERE / "portfolio_per_stock.json"
    serializable = {}
    for tk, w in portfolio.items():
        m_clean = {k: (v if v != float("inf") else "inf") for k, v in w["metrics"].items()
                   if k != "per_stock"}
        serializable[tk] = {
            "ticker": tk,
            "tf": w["tf"],
            "strategy": w["strategy"],
            "seed": w["seed"],
            "score": w["score"],
            "metrics": m_clean,
            "params": w["params"],
        }
    out.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\nSaved portfolio → {out}")
    print(f"Total time: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
