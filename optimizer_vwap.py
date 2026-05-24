"""
VWAP Pullback intraday strategy optimizer.

Same guided random-search algorithm as optimizer.py but with VWAP Pullback
strategy logic instead of MA crossover. Designed for 5m/10m/15m intraday
backtests of liquid US stocks.

STRATEGY:
    Regime (continuous):
      - VWAP rising over last N bars
      - EMA > VWAP (optional)
      - In session window (skip first 3 bars of day)

    Trigger:
      - Within last K bars, low touched VWAP (within tolerance %)
      - Current bar closes back above VWAP AND is bullish (c > o, c > prev c)
      - Volume above MA × multiplier (optional)

    Exits:
      - ATR-based stop OR R-multiple take-profit (OCO)
      - Optional: close < VWAP regime break
      - Forced close at last bar of each session day
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

# Reuse from the MA-cross optimizer
from optimizer import (
    TICKERS, HERE, DATA_DIR, TARGETS,
    MIN_TRADES_PER_STOCK, COMMISSION_PER_SIDE,
    get_data, _metrics,
)


# ═════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═════════════════════════════════════════════════════════════════════════════
def _vwap_intraday(df: pd.DataFrame) -> np.ndarray:
    """Cumulative VWAP that resets each calendar day (US market convention)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = (typical * df["Volume"]).astype(float)
    vol = df["Volume"].astype(float)

    # Day key based on index date
    if hasattr(df.index, "tz") and df.index.tz is not None:
        # If timezone-aware, group by local date
        date_key = df.index.tz_convert("America/New_York").date if str(df.index.tz) != "America/New_York" else df.index.date
    else:
        date_key = df.index.date
    date_key = np.asarray(date_key)

    # Vectorized cumsum-by-group
    pv_arr = pv.values
    vol_arr = vol.values
    out_pv = np.empty(len(pv_arr))
    out_vol = np.empty(len(vol_arr))
    running_pv = 0.0
    running_vol = 0.0
    prev_day = None
    for i in range(len(pv_arr)):
        if date_key[i] != prev_day:
            running_pv = 0.0
            running_vol = 0.0
            prev_day = date_key[i]
        running_pv += pv_arr[i]
        running_vol += vol_arr[i]
        out_pv[i] = running_pv
        out_vol[i] = running_vol

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(out_vol > 0, out_pv / out_vol, np.nan)


def _atr(df: pd.DataFrame, n: int) -> np.ndarray:
    h = df["High"].astype(float).values
    l = df["Low"].astype(float).values
    c = df["Close"].astype(float).values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - prev_c),
        np.abs(l - prev_c),
    ])
    # Wilder smoothing via EWM
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().values


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
def backtest_vwap(df: pd.DataFrame, p: dict) -> tuple[list, dict]:
    if df is None or len(df) < 80:
        return [], _metrics([])

    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    open_ = df["Open"].astype(float).values
    volume = df["Volume"].astype(float).values

    n = len(close)

    vwap = _vwap_intraday(df)
    ema = df["Close"].astype(float).ewm(span=p["ema_len"], adjust=False).mean().values
    atr = _atr(df, p["atr_len"])
    vol_ma = pd.Series(volume).rolling(p["vol_ma_len"]).mean().values

    # Regime
    rl = p["vwap_rising_lookback"]
    vwap_lag = np.roll(vwap, rl)
    vwap_lag[:rl] = np.nan
    vwap_rising = vwap > vwap_lag

    ema_above = ema > vwap if p["require_ema_above_vwap"] else np.ones(n, dtype=bool)

    vol_ok = (volume > vol_ma * p["vol_mult"]) if p["use_vol_filter"] else np.ones(n, dtype=bool)

    regime = vwap_rising & ema_above

    # Session boundary: bar index of day position
    if hasattr(df.index, "tz") and df.index.tz is not None:
        try:
            dates = df.index.tz_convert("America/New_York").date
        except Exception:
            dates = df.index.date
    else:
        dates = df.index.date
    dates = np.asarray(dates)

    # Skip first N bars per session (volatility / VWAP unstable)
    skip_first = p["skip_first_bars"]
    same_day_count = np.zeros(n, dtype=int)
    for i in range(1, n):
        same_day_count[i] = same_day_count[i - 1] + 1 if dates[i] == dates[i - 1] else 0
    in_session_ok = same_day_count >= skip_first

    # State machine
    in_pos = False
    entry_px = 0.0
    stop_px = 0.0
    tp_px = 0.0
    trades: list[tuple[float, float, str]] = []

    pb_lookback = p["pullback_lookback"]
    tol_factor = 1.0 + p["vwap_touch_tolerance"] / 100.0

    start = max(p["ema_len"], p["atr_len"], p["vol_ma_len"], rl, pb_lookback) + 2

    for i in range(start, n):
        if np.isnan(vwap[i]) or np.isnan(ema[i]) or np.isnan(atr[i]):
            continue

        c, h, l, o = close[i], high[i], low[i], open_[i]

        # Force close at last bar of session
        if in_pos and (i == n - 1 or dates[i + 1] != dates[i]):
            trades.append((entry_px, c, "eod"))
            in_pos = False
            continue

        if in_pos:
            # Stop or TP hit intra-bar
            if l <= stop_px:
                trades.append((entry_px, stop_px, "stop"))
                in_pos = False
                continue
            if h >= tp_px:
                trades.append((entry_px, tp_px, "tp"))
                in_pos = False
                continue
            # VWAP loss exit
            if p["exit_on_vwap_loss"] and c < vwap[i]:
                trades.append((entry_px, c, "vwap_loss"))
                in_pos = False
                continue
        else:
            if not regime[i] or not vol_ok[i] or not in_session_ok[i]:
                continue

            # Pullback: did a recent bar touch VWAP?
            recent_touch = False
            for j in range(1, pb_lookback + 1):
                k = i - j
                if k < 0:
                    break
                if dates[k] != dates[i]:
                    break  # don't carry pullback across sessions
                if low[k] <= vwap[k] * tol_factor:
                    recent_touch = True
                    break
            if not recent_touch:
                continue

            # Bounce confirmation
            bounce = c > vwap[i] and c > o and c > close[i - 1]
            if not bounce:
                continue

            in_pos = True
            entry_px = c
            stop_dist = atr[i] * p["init_stop_atr"]
            stop_px = c - stop_dist
            tp_px = c + stop_dist * p["take_profit_r"]

    if in_pos:
        trades.append((entry_px, close[-1], "eod"))

    return trades, _metrics(trades)


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
def evaluate(params: dict, data: dict[str, pd.DataFrame]) -> dict:
    per_stock = {}
    for tk in TICKERS:
        df = data.get(tk)
        if df is None or len(df) < 80:
            per_stock[tk] = {"pnl_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                            "n_trades": 0, "best": 0.0, "worst": 0.0}
            continue
        _, m = backtest_vwap(df, params)
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


def score(m: dict) -> float:
    per_stock = m["per_stock"]
    min_trades_intraday = 5
    n_qualifying = sum(1 for ms in per_stock.values() if ms["n_trades"] >= min_trades_intraday)
    if n_qualifying < len(TICKERS):
        return -1e6 + n_qualifying * 1000 + min(m["pnl_avg"], 100)

    n_negative = sum(1 for ms in per_stock.values() if ms["pnl_pct"] < 0)
    pnl_term = m["pnl_avg"] / TARGETS["pnl_pct"]
    win_term = m["win_avg"] / TARGETS["win_rate"]
    pf_term = min(m["pf_avg"], 5.0) / TARGETS["profit_factor"]
    base = pnl_term * win_term * pf_term
    base *= max(0.1, 1.0 - 0.15 * n_negative)

    # Cliff: hitting all 3 targets ALWAYS beats missing any
    if hits_target(m):
        return 1000.0 + base
    return base


def hits_target(m: dict) -> bool:
    return (m["pnl_avg"] >= TARGETS["pnl_pct"]
            and m["win_avg"] >= TARGETS["win_rate"]
            and m["pf_avg"] >= TARGETS["profit_factor"])


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETER GRID (VWAP-pullback specific)
# ═════════════════════════════════════════════════════════════════════════════
VWAP_GRID = {
    "vwap_rising_lookback":   [5, 10, 15, 20, 30],
    "ema_len":                [10, 20, 50, 100],
    "require_ema_above_vwap": [True, False],
    "pullback_lookback":      [3, 5, 8, 12, 20],
    "vwap_touch_tolerance":   [0.05, 0.10, 0.20, 0.30, 0.50],
    "use_vol_filter":         [True, False],
    "vol_ma_len":             [10, 20, 30],
    "vol_mult":               [1.0, 1.2, 1.5, 2.0],
    "atr_len":                [7, 14, 21],
    "init_stop_atr":          [1.0, 1.5, 2.0, 2.5],
    "take_profit_r":          [1.5, 2.0, 2.5, 3.0, 4.0],
    "exit_on_vwap_loss":      [True, False],
    "skip_first_bars":        [0, 3, 6, 12],
}


def rand_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in VWAP_GRID.items()}


def mutate(base: dict, rng: random.Random) -> dict:
    p = dict(base)
    fields = rng.sample(list(VWAP_GRID.keys()), k=rng.choice([1, 1, 2]))
    for f in fields:
        p[f] = rng.choice(VWAP_GRID[f])
    return p


# ═════════════════════════════════════════════════════════════════════════════
# OPTIMIZE / REPORT
# ═════════════════════════════════════════════════════════════════════════════
def optimize(data: dict[str, pd.DataFrame], n_iters: int = 400, seed: int = 42) -> dict:
    rng = random.Random(seed)
    best = None
    best_s = -float("inf")
    target_first_hit = None
    history_size = 0

    print(f"Optimizing VWAP Pullback — targets: PnL ≥ {TARGETS['pnl_pct']}%  Win ≥ {TARGETS['win_rate']:.0%}  PF ≥ {TARGETS['profit_factor']}")
    print(f"Budget: {n_iters} iterations")
    print("─" * 90)

    for i in range(n_iters):
        if best is not None and rng.random() < 0.30:
            params = mutate(best["params"], rng)
        else:
            params = rand_params(rng)

        m = evaluate(params, data)
        s = score(m)
        history_size += 1

        if s > best_s:
            best_s = s
            best = {"params": params, "metrics": m, "score": s}
            print(f"iter {i+1:>4}  ★  PnL {m['pnl_avg']:+7.1f}%  "
                  f"Win {m['win_avg']*100:5.1f}%  PF {m['pf_avg']:5.2f}  N {m['n_total']:>4}")

        if s >= 1000.0 and target_first_hit is None:
            target_first_hit = i + 1
            print(f"\n>>> TARGETS MET at iter {i+1} <<<\n")

    if best is None:
        best = {"params": {}, "metrics": evaluate(rand_params(rng), data), "score": -float("inf")}
    best["history_size"] = history_size
    best["target_first_hit"] = target_first_hit
    return best


def report(best: dict) -> None:
    m = best["metrics"]
    print()
    print("═" * 90)
    print("BEST RESULT (VWAP Pullback)")
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
        print(f"  {tk:>6}  {ms['pnl_pct']:+8.1f}  {ms['win_rate']*100:4.0f}%  "
              f"{pf_disp:>5}  {ms['n_trades']:>4}  {ms['best']*100:+6.1f}%  {ms['worst']*100:+6.1f}%")
    print()
    print("Parameters:")
    for k, v in best["params"].items():
        print(f"  {k:>22}: {v}")
    print()


def run_for_timeframe(tf: str, n_iters: int = 400, seed: int = 42) -> dict:
    print(f"\n{'#' * 90}\n# VWAP Pullback @ {tf}\n{'#' * 90}\n")
    t0 = time.time()
    data = get_data(tf)
    for tk in TICKERS:
        df = data.get(tk)
        if df is None or len(df) == 0:
            print(f"  {tk:>6}: NO DATA")
        else:
            print(f"  {tk:>6}: {len(df):>5} bars")

    best = optimize(data, n_iters=n_iters, seed=seed)
    report(best)
    out = HERE / f"best_params_vwap_{tf}.json"
    out.write_text(json.dumps(best, indent=2, default=str))
    print(f"Saved → {out}  ({time.time() - t0:.1f}s)")
    return best


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", nargs="*", default=["5m", "10m", "15m"])
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--pnl-target", type=float, default=None)
    parser.add_argument("--win-target", type=float, default=None)
    parser.add_argument("--pf-target",  type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    args = parser.parse_args()

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

    print("\n" + "═" * 90)
    print("CROSS-TIMEFRAME SUMMARY (VWAP Pullback)")
    print("═" * 90)
    print(f"{'TF':>5}  {'PnL %':>8}  {'Win %':>6}  {'PF':>6}  {'Trades':>7}  {'Target?':>8}")
    for tf, b in summary.items():
        m = b["metrics"]
        ok = "YES" if hits_target(m) else "no"
        print(f"  {tf:>3}  {m['pnl_avg']:+8.1f}  {m['win_avg']*100:5.1f}  {m['pf_avg']:6.2f}  {m['n_total']:>7}  {ok:>8}")

    print(f"\nTotal time: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
