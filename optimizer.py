"""
Optimization-based backtester for the MA-crossover + slope filter + trailing-% strategy.

Replicates the Pine v6 logic from slop_trial_check.pine, then searches the parameter
space using guided random search (random + mutate-around-best, with elitism — a simple
genetic-style optimization) until aggregate targets are hit OR the iteration budget
is exhausted.

Targets:
    Net PnL (avg across tickers)  ≥ 35%
    Win Rate (avg across tickers) ≥ 60%
    Profit Factor (avg across t.)  ≥ 1.5

Tickers: TSLA NVDA AAPL NFLX AMZN PLTR SOFI HOOD
Data:    daily OHLCV from yfinance (free), up to 5 years (less if IPO'd later)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
TICKERS = ["TSLA", "NVDA", "AAPL", "NFLX", "AMZN", "PLTR", "SOFI", "HOOD"]
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

TARGETS = {"pnl_pct": 35.0, "win_rate": 0.60, "profit_factor": 1.5}
MIN_TRADES_PER_STOCK = 5  # lowered for intraday (60-day window); was 10 for daily
COMMISSION_PER_SIDE = 0.001  # 0.1% per side -> 0.2% round trip


def _cache_path(tf: str) -> Path:
    return DATA_DIR / f"ohlcv_{tf}.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Data ranges per timeframe (yfinance limits)
TF_HISTORY = {
    "5m":  pd.Timedelta(days=59),    # yfinance allows up to 60 days for 5m
    "15m": pd.Timedelta(days=59),    # same limit
    "1h":  pd.Timedelta(days=729),   # yfinance allows up to 730 days for 1h
    "1d":  pd.Timedelta(days=5 * 365 + 30),  # essentially unlimited
}

# yfinance native intervals (10m is NOT native — we resample 5m → 10m)
NATIVE = {"5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}


def get_data(tf: str = "1d") -> dict[str, pd.DataFrame]:
    """
    Download OHLCV per ticker at given timeframe. Caches per-tf in a pickle file.
    Supports native yfinance intervals; for 10m we synthesize via resample from 5m.
    """
    if tf == "10m":
        # Synthesize from 5m
        five = get_data("5m")
        out = {}
        for tk, df in five.items():
            if df is None or len(df) == 0:
                out[tk] = df
                continue
            r = df.resample("10min").agg(
                {"Open": "first", "High": "max", "Low": "min",
                 "Close": "last", "Volume": "sum"}
            ).dropna()
            out[tk] = r
        return out

    cache = _cache_path(tf)
    if cache.exists():
        try:
            cached = pd.read_pickle(cache)
            tickers_in = cached.columns.get_level_values(0).unique().tolist()
            if all(t in tickers_in for t in TICKERS):
                return {t: cached[t].dropna() for t in TICKERS}
        except Exception:
            pass

    interval = NATIVE[tf]
    end = pd.Timestamp.now(tz="UTC")
    start = end - TF_HISTORY[tf]
    print(f"Downloading {len(TICKERS)} tickers @ {tf} from {start.date()} to {end.date()}...")
    raw = yf.download(
        TICKERS,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.to_pickle(cache)
        return {t: raw[t].dropna() for t in TICKERS}
    else:
        only = TICKERS[0]
        return {only: raw.dropna()}


# ─────────────────────────────────────────────────────────────────────────────
def _ma(series: pd.Series, n: int, kind: str) -> pd.Series:
    if kind == "EMA":
        return series.ewm(span=n, adjust=False).mean()
    return series.rolling(n).mean()


def backtest_one(df: pd.DataFrame, p: dict) -> tuple[list[tuple[float, float, str]], dict]:
    """
    Replicate the Pine strategy on one ticker.
    Returns list of (entry_price, exit_price, exit_reason) and a metrics dict.
    """
    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values

    n = len(close)
    fast = _ma(df["Close"].astype(float), p["fast_len"], p["ma_type"]).values
    slow = _ma(df["Close"].astype(float), p["slow_len"], p["ma_type"]).values

    sl = p["slope_len"]
    # safe slope: pct change of MA over sl bars
    fast_lag = np.roll(fast, sl)
    slow_lag = np.roll(slow, sl)
    fast_lag[:sl] = np.nan
    slow_lag[:sl] = np.nan
    fast_slope = (fast - fast_lag) / fast_lag * 100.0
    slow_slope = (slow - slow_lag) / slow_lag * 100.0

    if p["use_slope"]:
        slope_ok = (np.abs(fast_slope) >= p["slope_min"]) & (np.abs(slow_slope) >= p["slope_min"])
    else:
        slope_ok = np.ones(n, dtype=bool)

    # Cross signals
    cross_up = np.zeros(n, dtype=bool)
    cross_down = np.zeros(n, dtype=bool)
    cross_up[1:] = (fast[1:] > slow[1:]) & (fast[:-1] <= slow[:-1])
    cross_down[1:] = (fast[1:] < slow[1:]) & (fast[:-1] >= slow[:-1])

    buy_signal = cross_up & slope_ok
    sell_signal = cross_down  # exit not gated by slope

    # State machine
    in_pos = False
    entry_px = 0.0
    highest = 0.0
    trail_dist = 0.0
    trades: list[tuple[float, float, str]] = []

    for i in range(n):
        if np.isnan(fast[i]) or np.isnan(slow[i]):
            continue
        c, h, l = close[i], high[i], low[i]

        if in_pos:
            if h > highest:
                highest = h

            if p["use_trail"]:
                trail_stop = highest - trail_dist
                if l <= trail_stop:
                    trades.append((entry_px, trail_stop, "trail"))
                    in_pos = False
                    continue

            if sell_signal[i]:
                trades.append((entry_px, c, "cross"))
                in_pos = False
                continue

        else:
            if buy_signal[i]:
                in_pos = True
                entry_px = c
                highest = h
                trail_dist = entry_px * p["trail_percent"] / 100.0 if p["use_trail"] else 0.0

    if in_pos:
        trades.append((entry_px, close[-1], "eod"))

    return trades, _metrics(trades)


def _metrics(trades: list[tuple[float, float, str]]) -> dict:
    if not trades:
        return {"pnl_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "n_trades": 0,
                "best": 0.0, "worst": 0.0}
    cap = 1.0
    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0
    rets = []
    for entry, ex, _ in trades:
        if entry <= 0:
            continue
        # Apply commission: cost on entry AND exit
        gross_r = (ex - entry) / entry
        r = gross_r - 2 * COMMISSION_PER_SIDE
        rets.append(r)
        cap *= (1.0 + r)
        if r > 0:
            wins += 1
            gross_win += r
        else:
            losses += 1
            gross_loss += abs(r)
    n = len(trades)
    pnl_pct = (cap - 1.0) * 100.0
    win_rate = wins / n if n else 0.0
    if gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = float("inf") if gross_win > 0 else 0.0
    return {
        "pnl_pct": pnl_pct,
        "win_rate": win_rate,
        "profit_factor": pf,
        "n_trades": n,
        "best": max(rets) if rets else 0.0,
        "worst": min(rets) if rets else 0.0,
    }


def evaluate(params: dict, data: dict[str, pd.DataFrame]) -> dict:
    per_stock = {}
    for tk in TICKERS:
        df = data.get(tk)
        if df is None or len(df) < 100:
            per_stock[tk] = {"pnl_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "n_trades": 0,
                            "best": 0.0, "worst": 0.0}
            continue
        _, m = backtest_one(df, params)
        per_stock[tk] = m

    pnl_avg = float(np.mean([m["pnl_pct"] for m in per_stock.values()]))
    win_avg = float(np.mean([m["win_rate"] for m in per_stock.values()]))
    pfs = [min(m["profit_factor"], 10.0) for m in per_stock.values() if m["n_trades"] > 0]
    pf_avg = float(np.mean(pfs)) if pfs else 0.0
    n_total = sum(m["n_trades"] for m in per_stock.values())
    return {
        "pnl_avg": pnl_avg,
        "win_avg": win_avg,
        "pf_avg": pf_avg,
        "n_total": n_total,
        "per_stock": per_stock,
    }


def hits_target(m: dict) -> bool:
    return (
        m["pnl_avg"] >= TARGETS["pnl_pct"]
        and m["win_avg"] >= TARGETS["win_rate"]
        and m["pf_avg"] >= TARGETS["profit_factor"]
    )


def score(m: dict) -> float:
    """Composite ranking score (higher better). Heavily penalize sparse-trading combos.
    CLIFF: anything that hits all 3 targets always beats anything that doesn't."""
    per_stock = m["per_stock"]
    # Hard requirement: every stock must trade at least MIN_TRADES_PER_STOCK times
    n_qualifying = sum(1 for ms in per_stock.values() if ms["n_trades"] >= MIN_TRADES_PER_STOCK)
    if n_qualifying < len(TICKERS):
        return -1e6 + n_qualifying * 1000 + min(m["pnl_avg"], 100)

    n_negative = sum(1 for ms in per_stock.values() if ms["pnl_pct"] < 0)
    pnl_term = m["pnl_avg"] / TARGETS["pnl_pct"]
    win_term = m["win_avg"] / TARGETS["win_rate"]
    pf_term = min(m["pf_avg"], 5.0) / TARGETS["profit_factor"]
    base = pnl_term * win_term * pf_term
    base *= max(0.1, 1.0 - 0.15 * n_negative)

    # Cliff bonus: hitting all 3 targets ALWAYS beats missing any
    if hits_target(m):
        return 1000.0 + base
    return base


# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "fast_len":     [5, 7, 9, 12, 15, 20, 25],
    "slow_len":     [20, 25, 30, 40, 50, 80, 100, 150, 200],
    "ma_type":      ["EMA", "SMA"],
    "slope_len":    [2, 3, 5, 7, 10],
    "slope_min":    [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    "trail_percent": [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
    "use_slope":    [True, False],
    "use_trail":    [True, False],
}


def rand_params(rng: random.Random) -> dict:
    p = {k: rng.choice(v) for k, v in PARAM_GRID.items()}
    if p["fast_len"] >= p["slow_len"]:
        p["slow_len"] = rng.choice([x for x in PARAM_GRID["slow_len"] if x > p["fast_len"]])
    return p


def mutate(base: dict, rng: random.Random) -> dict:
    p = dict(base)
    fields = rng.sample(list(PARAM_GRID.keys()), k=rng.choice([1, 1, 2]))
    for f in fields:
        p[f] = rng.choice(PARAM_GRID[f])
    if p["fast_len"] >= p["slow_len"]:
        p["slow_len"] = rng.choice([x for x in PARAM_GRID["slow_len"] if x > p["fast_len"]])
    return p


# ─────────────────────────────────────────────────────────────────────────────
def optimize(data: dict[str, pd.DataFrame], n_iters: int = 400, seed: int = 42) -> dict:
    rng = random.Random(seed)
    best = None
    best_s = -float("inf")
    history = []
    target_first_hit = None

    print(f"Optimizing — targets: PnL ≥ {TARGETS['pnl_pct']}%  Win ≥ {TARGETS['win_rate']:.0%}  PF ≥ {TARGETS['profit_factor']}")
    print(f"Budget: {n_iters} iterations  |  search = random + mutate-around-best")
    print("─" * 90)

    for i in range(n_iters):
        # 30% mutate around best (once we have one), 70% explore
        if best is not None and rng.random() < 0.30:
            params = mutate(best["params"], rng)
        else:
            params = rand_params(rng)

        m = evaluate(params, data)
        s = score(m)
        history.append({"params": params, "metrics": m, "score": s})

        improved = s > best_s
        if improved:
            best_s = s
            best = {"params": params, "metrics": m, "score": s}
            print(
                f"iter {i+1:>4}  ★  PnL {m['pnl_avg']:+7.1f}%  "
                f"Win {m['win_avg']*100:5.1f}%  PF {m['pf_avg']:5.2f}  N {m['n_total']:>4}"
            )

        # Only mark "targets met" when the combo also passes the trade-count gate
        # (so the message matches what score() actually rewards via the cliff bonus)
        if s >= 1000.0 and target_first_hit is None:
            target_first_hit = i + 1
            print(f"\n>>> TARGETS MET at iter {i+1} — keeping search for better <<<\n")

    best["history_size"] = len(history)
    best["target_first_hit"] = target_first_hit
    return best


# ─────────────────────────────────────────────────────────────────────────────
def report(best: dict) -> None:
    m = best["metrics"]
    print()
    print("═" * 90)
    print("BEST RESULT")
    print("═" * 90)
    print(f"Aggregate (avg across 8 tickers)")
    print(f"  PnL %:          {m['pnl_avg']:+.2f}")
    print(f"  Win rate:       {m['win_avg']:.1%}")
    print(f"  Profit factor:  {m['pf_avg']:.2f}")
    print(f"  Total trades:   {m['n_total']}")
    print(f"  Score:          {best['score']:.3f}")
    print()
    print("Per-stock breakdown:")
    print(f"  {'Ticker':>6}  {'PnL %':>8}  {'Win':>5}  {'PF':>5}  {'N':>4}  {'Best':>7}  {'Worst':>7}")
    for tk, ms in m["per_stock"].items():
        pf_disp = f"{ms['profit_factor']:.2f}" if ms['profit_factor'] != float('inf') else "  inf"
        print(
            f"  {tk:>6}  {ms['pnl_pct']:+8.1f}  {ms['win_rate']*100:4.0f}%  "
            f"{pf_disp:>5}  {ms['n_trades']:>4}  {ms['best']*100:+6.1f}%  {ms['worst']*100:+6.1f}%"
        )
    print()
    print("Parameters:")
    for k, v in best["params"].items():
        print(f"  {k:>16}: {v}")
    print()
    if best.get("target_first_hit"):
        print(f"Targets first met at iteration {best['target_first_hit']} of {best['history_size']}.")
    else:
        gaps = []
        if m["pnl_avg"] < TARGETS["pnl_pct"]:
            gaps.append(f"PnL {m['pnl_avg']:.1f}% vs target {TARGETS['pnl_pct']}%")
        if m["win_avg"] < TARGETS["win_rate"]:
            gaps.append(f"Win {m['win_avg']:.1%} vs target {TARGETS['win_rate']:.0%}")
        if m["pf_avg"] < TARGETS["profit_factor"]:
            gaps.append(f"PF {m['pf_avg']:.2f} vs target {TARGETS['profit_factor']}")
        print("Targets NOT fully met. Gaps: " + "; ".join(gaps))


def run_for_timeframe(tf: str, n_iters: int = 600, seed: int = 42) -> dict:
    print(f"\n{'#' * 90}\n# TIMEFRAME: {tf}\n{'#' * 90}\n")
    t0 = time.time()
    data = get_data(tf)
    print(f"Bars per ticker:")
    for tk in TICKERS:
        df = data.get(tk)
        if df is None or len(df) == 0:
            print(f"  {tk:>6}: NO DATA")
        else:
            first = df.index[0]
            last = df.index[-1]
            # Intraday index includes time; daily is just date
            try:
                first_s = first.strftime("%Y-%m-%d %H:%M") if tf != "1d" else first.strftime("%Y-%m-%d")
                last_s  = last.strftime("%Y-%m-%d %H:%M")  if tf != "1d" else last.strftime("%Y-%m-%d")
            except Exception:
                first_s, last_s = str(first), str(last)
            print(f"  {tk:>6}: {len(df):>5} bars  {first_s} → {last_s}")

    best = optimize(data, n_iters=n_iters, seed=seed)
    report(best)

    out = HERE / f"best_params_{tf}.json"
    out.write_text(json.dumps(best, indent=2, default=str))
    print(f"\nSaved → {out}")
    print(f"Time for {tf}: {time.time() - t0:.1f}s")
    return best


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", nargs="*", default=["1d"],
                        help="Timeframes to run: 5m 10m 15m 1h 1d")
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--pnl-target", type=float, default=None, help="Override PnL target %")
    parser.add_argument("--win-target", type=float, default=None, help="Override win-rate target (0-1)")
    parser.add_argument("--pf-target",  type=float, default=None, help="Override profit-factor target")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42], help="Search seeds (try multiple to escape local optima)")
    args = parser.parse_args()

    # Apply target overrides
    if args.pnl_target is not None: TARGETS["pnl_pct"] = args.pnl_target
    if args.win_target is not None: TARGETS["win_rate"] = args.win_target
    if args.pf_target  is not None: TARGETS["profit_factor"] = args.pf_target

    t = time.time()
    summary = {}
    for tf in args.tf:
        tf_best = None
        for seed in args.seeds:
            print(f"\n--- seed={seed} ---")
            cand = run_for_timeframe(tf, n_iters=args.iters, seed=seed)
            if tf_best is None or cand["score"] > tf_best["score"]:
                tf_best = cand
        summary[tf] = tf_best

    # Cross-timeframe summary
    print("\n" + "═" * 90)
    print("CROSS-TIMEFRAME SUMMARY")
    print("═" * 90)
    print(f"{'TF':>5}  {'PnL %':>8}  {'Win %':>6}  {'PF':>6}  {'Trades':>7}  {'Target?':>8}")
    for tf, b in summary.items():
        m = b["metrics"]
        ok = "YES" if (m["pnl_avg"] >= TARGETS["pnl_pct"]
                       and m["win_avg"] >= TARGETS["win_rate"]
                       and m["pf_avg"] >= TARGETS["profit_factor"]) else "no"
        print(f"  {tf:>3}  {m['pnl_avg']:+8.1f}  {m['win_avg']*100:5.1f}  {m['pf_avg']:6.2f}  {m['n_total']:>7}  {ok:>8}")

    print(f"\nTotal time: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
