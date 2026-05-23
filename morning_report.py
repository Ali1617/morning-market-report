"""
Morning Market Report
Runs every weekday at 9:30 AM ET — via GitHub Actions (cloud) or Windows Task Scheduler (local)
Scans S&P 500 and NDX 100 for EMA 9/21 strategy signals + news
Cloud mode: sends HTML report by email (set GMAIL_USER, GMAIL_APP_PASSWORD, REPORT_EMAIL env vars)
Local mode: saves HTML report and opens it in the browser
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os, sys, webbrowser, warnings, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FAST_LEN, SLOW_LEN, MA_TYPE = 9, 21, "EMA"
TRAIL_PCT   = 1.5
SLOPE_LEN   = 3
SLOPE_MIN   = 0.05
TOP_N       = 10
OUTPUT_DIR  = r"D:\Claude\reports"

# Historical avg win from backtest (used for expected profit estimate)
HIST_AVG_WIN = 2.5   # %

# ─── NDX 100 COMPONENTS ──────────────────────────────────────────────────────
NDX100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AMD","ADBE","CSCO","QCOM","TXN","INTU","AMAT","MU",
    "LRCX","KLAC","MRVL","PANW","SNPS","CDNS","CRWD","FTNT","MNST","MDLZ",
    "PEP","SBUX","GILD","AMGN","VRTX","REGN","IDXX","MRNA","PYPL","ADSK",
    "WDAY","TEAM","DDOG","ZS","PLTR","MELI","ABNB","TTD","PCAR","HON",
    "EA","CMCSA","CHTR","TMUS","CSX","FAST","CTAS","PAYX","ROP","GEHC",
    "LULU","ROST","CPRT","MCHP","CDW","CEG","FSLR","BKR","CTSH","NXPI",
    "ON","INTC","KDP","AZN","BIIB","ILMN","COIN","HOOD","RIVN","SMCI",
    "ODFL","VRSK","XEL","AEP","EXC","GFS","DLTR","MTCH","ZM","RIVN",
]

# ─── S&P 500 — fetch live from Wikipedia, fall back to top-100 ───────────────
SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","BRK-B","JPM",
    "LLY","V","XOM","UNH","MA","JNJ","PG","HD","COST","MRK","ABBV","CVX",
    "KO","BAC","PEP","ADBE","TMO","CSCO","CRM","ACN","MCD","ABT","NFLX",
    "WMT","DHR","LIN","DIS","AMD","TXN","NEE","PM","ORCL","AMGN","INTU",
    "RTX","QCOM","SPGI","UNP","HON","MS","GE","IBM","CAT","AMAT","ISRG",
    "BLK","GS","ELV","MDLZ","ADP","MMC","T","SYK","PLD","DE","BKNG","VRTX",
    "ADI","REGN","ZTS","CB","C","NOW","GILD","PANW","SBUX","MU","CI","KLAC",
    "SO","DUK","AON","CME","BSX","NOC","ETN","ITW","APD","FDX","ECL","SHW",
    "WM","MCO","USB","HUM","LRCX","NSC","PGR","FCX","INTC","EW","MAR","MCK",
]

def get_sp500_tickers():
    try:
        import requests
        from io import StringIO
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15
        ).text
        df = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"    Loaded {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"    Wikipedia fetch failed ({e}), using fallback list")
        return SP500_FALLBACK

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def calc_ma(s, n, t="EMA"):
    return s.ewm(span=n, adjust=False).mean() if t == "EMA" else s.rolling(n).mean()

def resample_10m(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df[["Open","High","Low","Close","Volume"]].resample("10min").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna()

def score_ticker(df):
    """Score a ticker for signal strength. Returns None if not scoreable."""
    if len(df) < SLOW_LEN + SLOPE_LEN + 10:
        return None

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    volume = df["Volume"].squeeze()

    fast = calc_ma(close, FAST_LEN)
    slow = calc_ma(close, SLOW_LEN)

    fs = (fast - fast.shift(SLOPE_LEN)) / fast.shift(SLOPE_LEN) * 100
    ss = (slow - slow.shift(SLOPE_LEN)) / slow.shift(SLOPE_LEN) * 100

    slope_ok = (fs.abs() >= SLOPE_MIN) & (ss.abs() >= SLOPE_MIN)
    above    = fast.iloc[-1] > slow.iloc[-1]

    # Crossover events
    crossover   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    crossunder  = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    # Bars since last buy crossover (look back 20 bars)
    bars_since_cross = None
    for i in range(1, 21):
        if i <= len(crossover) and crossover.iloc[-i]:
            bars_since_cross = i
            break

    # Was there a sell signal after last buy? (invalidates signal)
    signal_valid = True
    if bars_since_cross:
        for i in range(1, bars_since_cross):
            if crossunder.iloc[-i]:
                signal_valid = False
                break

    ma_gap_pct  = (close.iloc[-1] - fast.iloc[-1]) / fast.iloc[-1] * 100

    # Volume surge (vs 20-bar avg)
    avg_vol = volume.iloc[-21:-1].mean()
    vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1.0

    # ── Scoring ──
    score = 0
    if above:           score += 30
    if slope_ok.iloc[-1]: score += 20
    if bars_since_cross and signal_valid:
        freshness = max(0, 15 - bars_since_cross)
        score += 15 + freshness
    if fs.iloc[-1] > SLOPE_MIN * 3:  score += 10   # strong upslope
    if ma_gap_pct > 0:               score += min(8, ma_gap_pct * 2)
    if vol_ratio > 1.5:              score += 5
    if not signal_valid:             score -= 20

    # Day-of change
    prev_close = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
    day_chg    = (close.iloc[-1] - prev_close) / prev_close * 100

    return {
        "score":            round(score, 1),
        "in_trend":         bool(above),
        "slope_ok":         bool(slope_ok.iloc[-1]),
        "bars_since_cross": bars_since_cross,
        "signal_valid":     signal_valid,
        "fast_slope":       round(float(fs.iloc[-1]), 3),
        "slow_slope":       round(float(ss.iloc[-1]), 3),
        "ma_gap_pct":       round(float(ma_gap_pct), 3),
        "vol_ratio":        round(float(vol_ratio), 2),
        "price":            round(float(close.iloc[-1]), 2),
        "fast_ma":          round(float(fast.iloc[-1]), 2),
        "slow_ma":          round(float(slow.iloc[-1]), 2),
        "day_chg":          round(float(day_chg), 2),
    }

def get_ticker_news(ticker, max_items=3):
    try:
        t = yf.Ticker(ticker)
        raw = t.news or []
        items = []
        for n in raw[:max_items]:
            content = n.get("content", {})
            title   = content.get("title") or n.get("title", "")
            pub     = content.get("pubDate", "")
            if title:
                items.append({"title": title, "pub": pub})
        return items
    except:
        return []

def get_market_news():
    items = []
    for sym in ["^GSPC", "^NDX", "SPY", "QQQ", "^VIX"]:
        try:
            for n in (yf.Ticker(sym).news or [])[:3]:
                content = n.get("content", {})
                title   = content.get("title") or n.get("title", "")
                if title and title not in items:
                    items.append(title)
        except:
            pass
    return items[:10]

def get_index_snapshot():
    snap = {}
    for label, sym in [("S&P 500","^GSPC"),("Nasdaq 100","^NDX"),("VIX","^VIX")]:
        try:
            info = yf.Ticker(sym).fast_info
            snap[label] = {
                "price":    round(info.last_price, 2),
                "prev":     round(info.previous_close, 2),
                "chg_pct":  round((info.last_price - info.previous_close) / info.previous_close * 100, 2),
            }
        except:
            snap[label] = None
    return snap

# ─── SCANNER ─────────────────────────────────────────────────────────────────
def scan_universe(tickers, label):
    print(f"\n  Scanning {label} ({len(tickers)} tickers)...")
    results = []
    chunk = 50

    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i+chunk]
        pct   = int((i / len(tickers)) * 100)
        print(f"    {pct:>3}%  downloading {batch[0]}..{batch[-1]}", end="\r")
        try:
            raw = yf.download(
                batch, period="5d", interval="5m",
                auto_adjust=True, progress=False, group_by="ticker",
                threads=True
            )
            for tk in batch:
                try:
                    df_tk = raw[tk] if len(batch) > 1 else raw
                    if df_tk is None or df_tk.empty: continue
                    df_tk = df_tk.dropna()
                    if len(df_tk) < 30: continue
                    df10 = resample_10m(df_tk)
                    sig  = score_ticker(df10)
                    if sig and sig["score"] >= 40 and sig["in_trend"]:
                        sig["ticker"] = tk
                        results.append(sig)
                except:
                    pass
        except:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"    100%  done — {len(results)} valid signals found         ")
    return results[:TOP_N]

# ─── REPORT BUILDER ──────────────────────────────────────────────────────────
def expected_profit(score):
    """Estimate expected profit based on signal score."""
    base = HIST_AVG_WIN
    bonus = max(0, (score - 40) * 0.04)
    return round(min(base + bonus, 6.0), 1)

def render_cards(picks, title):
    if not picks:
        return f"<h2>{title}</h2><p style='color:#8b949e'>No strong signals found today.</p>"

    cards = ""
    for i, p in enumerate(picks):
        rank      = i + 1
        top3      = rank <= 3
        exp       = expected_profit(p["score"])
        cross_tag = ""
        if p["bars_since_cross"] and p["signal_valid"]:
            cross_tag = f'<span class="badge fresh">CROSS {p["bars_since_cross"]}b ago</span> '

        slope_col = "#3fb950" if p["fast_slope"] > 0.1 else "#f0883e"
        vol_tag   = '<span class="badge vol">VOL SURGE</span> ' if p["vol_ratio"] > 1.8 else ""
        chg_col   = "#3fb950" if p["day_chg"] >= 0 else "#f85149"

        news_html = ""
        for n in p.get("news", [])[:2]:
            title_text = n.get("title", "")[:100]
            if title_text:
                news_html += f'<div class="news-item">&#9656; {title_text}</div>\n'

        cards += f"""
<div class="card {'top3' if top3 else ''}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span class="ticker">#{rank} {p['ticker']}</span>
    <span style="color:{chg_col};font-weight:bold">{p['day_chg']:+.2f}%</span>
  </div>
  <div class="price">${p['price']}  &nbsp;|&nbsp; Fast MA: {p['fast_ma']}  &nbsp;|&nbsp; Slow MA: {p['slow_ma']}</div>
  <div style="margin:6px 0">{cross_tag}{vol_tag}<span class="badge score">score {p['score']}</span></div>
  <div class="signal-row">
    <span>Fast slope: <b style="color:{slope_col}">{p['fast_slope']:+.3f}%</b></span>
    &nbsp;&nbsp;
    <span>MA gap: <b>{p['ma_gap_pct']:+.2f}%</b></span>
    &nbsp;&nbsp;
    <span>Vol x{p['vol_ratio']}</span>
  </div>
  <div class="expect">Expected move: <b>+{exp}%</b> &nbsp;|&nbsp; Stop: -1.5% trail</div>
  {news_html}
</div>"""

    return f"<h2>{title}</h2>\n<div class='grid'>{cards}</div>"

def build_report(sp500_picks, ndx_picks, market_news, snapshot):
    now   = datetime.now()
    today = now.strftime("%A, %B %d, %Y")
    time  = now.strftime("%I:%M %p ET")

    def snap_row(label, d):
        if not d: return ""
        col = "#3fb950" if d["chg_pct"] >= 0 else "#f85149"
        arrow = "▲" if d["chg_pct"] >= 0 else "▼"
        return f"<td><b>{label}</b></td><td>{d['price']:,.2f}</td><td style='color:{col}'>{arrow} {d['chg_pct']:+.2f}%</td>"

    snap_html = "<table><tr>"
    for lbl in ["S&P 500","Nasdaq 100","VIX"]:
        snap_html += snap_row(lbl, snapshot.get(lbl))
    snap_html += "</tr></table>"

    news_items = "".join(f"<li>{n}</li>" for n in market_news[:8]) if market_news else "<li>No major news items found.</li>"

    # Fetch news for each pick
    all_picks = sp500_picks + ndx_picks
    print(f"\n  Fetching news for {len(all_picks)} picks...")
    for p in all_picks:
        p["news"] = get_ticker_news(p["ticker"])

    sp500_html = render_cards(sp500_picks, f"S&P 500 &mdash; Top {TOP_N} Picks")
    ndx_html   = render_cards(ndx_picks,   f"Nasdaq 100 &mdash; Top {TOP_N} Picks")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Morning Report &mdash; {today}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#0d1117; color:#e6edf3; padding:24px; line-height:1.5 }}
  h1   {{ color:#58a6ff; font-size:1.6em; border-bottom:1px solid #30363d;
          padding-bottom:12px; margin-bottom:16px }}
  h2   {{ color:#79c0ff; font-size:1.2em; margin:28px 0 12px }}
  .meta {{ color:#8b949e; font-size:0.85em; margin-bottom:20px }}
  .snap {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
           padding:14px 18px; display:inline-block; margin-bottom:16px }}
  .snap table td {{ padding:4px 18px 4px 0; font-size:0.95em }}
  .ctx  {{ background:#161b22; border-left:3px solid #388bfd;
           padding:12px 16px; border-radius:4px; margin-bottom:20px }}
  .ctx ul {{ padding-left:20px; color:#adbac7; font-size:0.9em }}
  .ctx li {{ margin:4px 0 }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); gap:12px }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px }}
  .card.top3 {{ border-color:#3fb950 }}
  .ticker {{ font-size:1.35em; font-weight:bold; color:#3fb950 }}
  .price  {{ color:#8b949e; font-size:0.85em; margin:4px 0 }}
  .badge  {{ display:inline-block; font-size:0.72em; padding:2px 8px;
             border-radius:12px; margin:2px; font-weight:600 }}
  .fresh  {{ background:#da3633; color:#fff }}
  .vol    {{ background:#9a6700; color:#fff }}
  .score  {{ background:#1f6feb; color:#fff }}
  .signal-row {{ font-size:0.85em; color:#adbac7; margin:6px 0 }}
  .expect {{ color:#3fb950; font-size:0.95em; margin:8px 0 4px }}
  .news-item {{ font-size:0.8em; color:#8b949e; border-left:2px solid #30363d;
                padding-left:8px; margin:3px 0 }}
  .risk {{ background:#161b22; border-left:3px solid #f0883e;
           padding:12px 16px; border-radius:4px }}
  .risk ul {{ padding-left:20px; color:#adbac7; font-size:0.9em }}
  .risk li {{ margin:4px 0 }}
  footer {{ color:#6e7681; font-size:0.75em; margin-top:40px; border-top:1px solid #21262d; padding-top:12px }}
</style>
</head>
<body>

<h1>&#9788; Morning Market Report &mdash; {today}</h1>
<p class="meta">Generated {time} &nbsp;|&nbsp; Strategy: EMA 9/21 &bull; 10-min bars &bull; Slope filter &bull; 1.5% trailing stop</p>

<h2>Index Snapshot</h2>
<div class="snap">{snap_html}</div>

<h2>Market &amp; Macro News</h2>
<div class="ctx"><ul>{news_items}</ul></div>

{sp500_html}

{ndx_html}

<h2>Risk Notes</h2>
<div class="risk">
  <ul>
    <li>Signals are based on last session's close — confirm the crossover is still intact at 9:30 open before entering</li>
    <li>Wait for the first 10-min bar to close before entering — never buy on the first candle</li>
    <li>Always use the 1.5% trailing stop, no exceptions</li>
    <li>On Fed days, CPI, NFP or major earnings — reduce size by 50% or skip entirely</li>
    <li>Expected move is a statistical average based on 60-day backtest — individual trades vary widely</li>
    <li>VIX above 25 = higher volatility, trail stop may be hit faster</li>
  </ul>
</div>

<footer>
  Auto-generated by Claude Code morning_report.py &nbsp;|&nbsp;
  EMA 9/21 + Slope Filter ({SLOPE_MIN}% min, {SLOPE_LEN}-bar lookback) + {TRAIL_PCT}% Trailing Stop
</footer>
</body>
</html>"""

# ─── EMAIL SENDER ────────────────────────────────────────────────────────────
def send_email(html: str, subject: str) -> bool:
    sender    = os.environ.get("GMAIL_USER", "")
    password  = os.environ.get("GMAIL_APP_PASSWORD", "")
    recipient = os.environ.get("REPORT_EMAIL", "")

    if not all([sender, password, recipient]):
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Morning Market Report <{sender}>"
    msg["To"]      = recipient
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
    return True

# ─── MAIN ────────────────────────────────────────────────────────────────────
CLOUD_MODE = bool(os.environ.get("GMAIL_USER"))   # True when running on GitHub Actions

def check_et_hour():
    """Exit early if it's not 9 AM ET — handles DST automatically."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.hour != 9:
            print(f"Not 9 AM ET (currently {now_et.strftime('%I:%M %p ET')}), skipping.")
            sys.exit(0)
    except Exception:
        pass   # if timezone check fails, run anyway

def main():
    if CLOUD_MODE:
        check_et_hour()

    print(f"\n{'='*62}")
    print(f"  MORNING REPORT  {datetime.now().strftime('%A %B %d, %Y  %I:%M %p')}")
    print(f"  Mode: {'CLOUD (email)' if CLOUD_MODE else 'LOCAL (browser)'}")
    print(f"{'='*62}")

    print("\n  [1/5] Fetching index snapshot...")
    snapshot = get_index_snapshot()
    for lbl, d in snapshot.items():
        if d:
            print(f"    {lbl}: {d['price']:>10,.2f}  ({d['chg_pct']:+.2f}%)")

    print("\n  [2/5] Fetching market news...")
    market_news = get_market_news()
    print(f"    {len(market_news)} items found")

    print("\n  [3/5] Loading S&P 500 components...")
    sp500 = get_sp500_tickers()
    sp500_picks = scan_universe(sp500, "S&P 500")

    print("\n  [4/5] Scanning NDX 100...")
    ndx_picks = scan_universe(NDX100, "NDX 100")

    print("\n  [5/5] Building report...")
    html = build_report(sp500_picks, ndx_picks, market_news, snapshot)

    today     = datetime.now().strftime("%Y%m%d")
    subject   = f"Morning Market Report — {datetime.now().strftime('%A %b %d')} | Top picks inside"

    if CLOUD_MODE:
        print("\n  Sending email...")
        try:
            send_email(html, subject)
            print(f"  Email sent to {os.environ['REPORT_EMAIL']}")
        except Exception as e:
            print(f"  Email FAILED: {e}")
            raise
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fname = os.path.join(OUTPUT_DIR, f"report_{today}.html")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  Saved: {fname}")
        webbrowser.open(f"file:///{fname}")
        print("  Report opened in browser.")

    print("\n  Top S&P 500 picks:")
    for p in sp500_picks[:5]:
        print(f"    {p['ticker']:<6}  score={p['score']:>5}  "
              f"slope={p['fast_slope']:+.3f}%  gap={p['ma_gap_pct']:+.2f}%  "
              f"exp=+{expected_profit(p['score'])}%")

    print("\n  Top NDX picks:")
    for p in ndx_picks[:5]:
        print(f"    {p['ticker']:<6}  score={p['score']:>5}  "
              f"slope={p['fast_slope']:+.3f}%  gap={p['ma_gap_pct']:+.2f}%  "
              f"exp=+{expected_profit(p['score'])}%")

    print(f"\n{'='*62}\n")

if __name__ == "__main__":
    main()
