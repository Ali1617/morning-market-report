"""
Replicates the PineScript "slop_trial_check" strategy in Python.
Runs across multiple tickers on 5m and 15m timeframes using yfinance data.

Install deps first:
    pip install yfinance pandas numpy
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — mirror your PineScript defaults
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL",
    "META", "AMZN", "SPY", "QQQ", "JPM",
    "NFLX", "AMD",  "INTC", "BA",   "DIS",
]

INTERVALS = ["5m", "15m"]

FAST_LEN   = 9
SLOW_LEN   = 21
MA_TYPE    = "EMA"    # "EMA" or "SMA"

USE_TRAIL   = True
TRAIL_PCT   = 1.5     # %

USE_SLOPE   = True
SLOPE_LEN   = 3       # bars
SLOPE_MIN   = 0.05    # % — minimum abs slope for both MAs


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def calc_ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    if ma_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    return series.rolling(length).mean()


def run_backtest(df: pd.DataFrame, ticker: str, interval: str) -> dict | None:
    if df is None or len(df) < SLOW_LEN + SLOPE_LEN + 10:
        return None

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()

    fast = calc_ma(close, FAST_LEN, MA_TYPE)
    slow = calc_ma(close, SLOW_LEN, MA_TYPE)

    # Slope filter — same formula as PineScript
    fast_slope = (fast - fast.shift(SLOPE_LEN)) / fast.shift(SLOPE_LEN) * 100
    slow_slope = (slow - slow.shift(SLOPE_LEN)) / slow.shift(SLOPE_LEN) * 100

    if USE_SLOPE:
        slope_ok = (fast_slope.abs() >= SLOPE_MIN) & (slow_slope.abs() >= SLOPE_MIN)
    else:
        slope_ok = pd.Series(True, index=close.index)

    # Crossover / crossunder
    buy_sig  = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    sell_sig = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    valid_buy  = buy_sig  & slope_ok
    valid_sell = sell_sig & slope_ok

    # ── Trade simulation ──────────────────────────────────────────────────────
    trades        = []
    in_trade      = False
    entry_price   = 0.0
    entry_idx     = 0
    highest_since = 0.0

    close_arr = close.values
    high_arr  = high.values
    buy_arr   = valid_buy.values
    sell_arr  = valid_sell.values

    for i in range(len(df)):
        c = float(close_arr[i])
        h = float(high_arr[i])

        if in_trade:
            highest_since = max(highest_since, h)
            trail_price   = highest_since * (1.0 - TRAIL_PCT / 100.0)

            trail_hit = USE_TRAIL and (c <= trail_price)
            ma_exit   = bool(sell_arr[i])

            if trail_hit or ma_exit:
                pnl_pct = (c - entry_price) / entry_price * 100.0
                trades.append({
                    "pnl_pct":     pnl_pct,
                    "exit_reason": "trail" if trail_hit else "ma_cross",
                    "bars_held":   i - entry_idx,
                })
                in_trade = False

        elif bool(buy_arr[i]):
            in_trade      = True
            entry_price   = c
            entry_idx     = i
            highest_since = h

    if not trades:
        return None

    t = pd.DataFrame(trades)
    wins   = t[t["pnl_pct"] > 0]
    losses = t[t["pnl_pct"] <= 0]

    win_rate      = len(wins) / len(t) * 100
    avg_win       = wins["pnl_pct"].mean()   if len(wins)   else 0.0
    avg_loss      = losses["pnl_pct"].mean() if len(losses) else 0.0
    gross_profit  = wins["pnl_pct"].sum()
    gross_loss    = losses["pnl_pct"].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf
    total_pnl     = t["pnl_pct"].sum()
    expectancy    = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss
    trail_exits   = int((t["exit_reason"] == "trail").sum())

    return {
        "Ticker":         ticker,
        "TF":             interval,
        "Trades":         len(t),
        "Win%":           round(win_rate, 1),
        "AvgWin%":        round(avg_win, 2),
        "AvgLoss%":       round(avg_loss, 2),
        "ProfitFactor":   round(profit_factor, 2) if profit_factor != np.inf else "∞",
        "Expectancy%":    round(expectancy, 3),
        "TotalPnL%":      round(total_pnl, 2),
        "AvgBars":        round(t["bars_held"].mean(), 1),
        "TrailExits":     trail_exits,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\nStrategy: {MA_TYPE} {FAST_LEN}/{SLOW_LEN}  |  "
          f"Slope filter: {USE_SLOPE} (len={SLOPE_LEN}, min={SLOPE_MIN}%)  |  "
          f"Trail stop: {USE_TRAIL} ({TRAIL_PCT}%)\n")
    print(f"{'Ticker':<6} {'TF':<4}  fetching...", end="", flush=True)

    results = []
    period  = "60d"   # yfinance max for intraday

    for ticker in TICKERS:
        for interval in INTERVALS:
            tag = f"{ticker} {interval}"
            print(f"\r  Downloading {tag:<12}", end="", flush=True)
            try:
                raw = yf.download(
                    ticker, period=period, interval=interval,
                    auto_adjust=True, progress=False, multi_level_index=False
                )
                if raw is None or raw.empty:
                    continue
                r = run_backtest(raw, ticker, interval)
                if r:
                    results.append(r)
            except Exception as exc:
                print(f"\n  {tag} — ERROR: {exc}")

    print("\r" + " " * 40 + "\r", end="")   # clear progress line

    if not results:
        print("No results generated — check your internet connection or ticker symbols.")
        return

    df = pd.DataFrame(results)

    # ── Summary by timeframe ──────────────────────────────────────────────────
    for tf in INTERVALS:
        sub = df[df["TF"] == tf].sort_values("TotalPnL%", ascending=False)
        if sub.empty:
            continue
        print(f"\n{'-'*95}")
        print(f"  {tf} RESULTS  ({len(sub)} symbols)")
        print(f"{'-'*95}")
        print(sub.to_string(index=False))

    # ── Overall aggregate ─────────────────────────────────────────────────────
    numeric_cols = ["Trades", "Win%", "AvgWin%", "AvgLoss%", "Expectancy%",
                    "TotalPnL%", "AvgBars", "TrailExits"]
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    print(f"\n{'-'*95}")
    print("  AGGREGATE ACROSS ALL SYMBOLS & TIMEFRAMES")
    print(f"{'-'*95}")
    agg = num_df.mean().round(2)
    print(f"  Avg Win Rate   : {agg['Win%']}%")
    print(f"  Avg Win        : {agg['AvgWin%']}%")
    print(f"  Avg Loss       : {agg['AvgLoss%']}%")
    print(f"  Avg Expectancy : {agg['Expectancy%']}% per trade")
    print(f"  Avg Total PnL  : {agg['TotalPnL%']}%  (sum of all trade PnLs, not compounded)")
    print(f"  Avg Bars Held  : {agg['AvgBars']}")
    print(f"  Trail exits    : {int(num_df['TrailExits'].sum())} / {int(num_df['Trades'].sum())} trades")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = "backtest_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
