#!/usr/bin/env python3
"""
Macro-Driven Morning Stock Report
-  Scrapes Trading Economics economic calendar (USD events)
-  Analyzes macro sentiment (bullish / bearish / neutral)
-  Selects top 10 long candidates from S&P 500 and NDX 100
   based on: macro tailwinds + pre-market momentum + news
-  Emails a full HTML report at 9 AM ET every weekday
No PineScript / backtest code used — pure fundamental + news selection.
"""

import os, sys, smtplib, warnings
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

warnings.filterwarnings("ignore")

CLOUD_MODE  = bool(os.environ.get("GMAIL_USER"))
OUTPUT_DIR  = r"D:\Claude\reports"
TOP_N       = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://tradingeconomics.com/",
}

# ─── STOCK UNIVERSE ───────────────────────────────────────────────────────────
SP500_SECTORS = {
    "Technology":            ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","CSCO","TXN","QCOM",
                              "AMD","MU","ADI","AMAT","KLAC","LRCX","IBM","HPQ","INTC","DELL"],
    "Consumer Discretionary":["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","MAR",
                              "HLT","RCL","CCL","F","GM","RIVN"],
    "Financials":            ["JPM","BAC","WFC","GS","MS","BLK","AXP","V","MA","SPGI",
                              "MCO","C","USB","CME","SCHW","COF"],
    "Healthcare":            ["UNH","LLY","JNJ","ABBV","MRK","TMO","ABT","DHR","BSX",
                              "ISRG","ELV","VRTX","REGN","CI","HUM"],
    "Industrials":           ["GE","CAT","RTX","HON","DE","BA","UNP","UPS","FDX","ETN",
                              "ITW","NOC","LMT","NSC","CSX","MMM"],
    "Energy":                ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","BKR","HAL","OXY"],
    "Communication":         ["META","GOOGL","NFLX","DIS","CMCSA","T","VZ","CHTR","EA","TTWO"],
    "Materials":             ["LIN","APD","SHW","FCX","NEM","ECL","ALB","NUE","DD","CF"],
    "Real Estate":           ["PLD","AMT","EQIX","WELL","PSA","SPG","O","VICI"],
    "Utilities":             ["NEE","DUK","SO","AEP","XEL","EXC","SRE","D"],
    "Consumer Staples":      ["WMT","PG","COST","KO","PEP","PM","MDLZ","CL","GIS","KHC"],
}

NDX_SECTORS = {
    "Semiconductors":        ["NVDA","AMD","AVGO","INTC","QCOM","TXN","MU","AMAT","KLAC",
                              "LRCX","SNPS","CDNS","MRVL","ON","MCHP","GFS","NXPI","SMCI"],
    "Mega Cap Tech":         ["AAPL","MSFT","GOOGL","AMZN","META","TSLA"],
    "Software / Cloud":      ["ADBE","INTU","CRWD","PANW","FTNT","ZS","DDOG","PLTR",
                              "WDAY","TEAM","ADSK","TTD","COIN","HOOD"],
    "Consumer / Retail":     ["COST","SBUX","MELI","ABNB","NFLX","BKNG"],
    "Biotech / Health":      ["GILD","AMGN","VRTX","REGN","MRNA","BIIB","IDXX"],
    "Industrials / Energy":  ["PCAR","HON","BKR","FSLR","CEG","CSX","FAST"],
    "Other":                 ["MNST","KDP","MDLZ","PEP","CHTR","CMCSA","EA"],
}

# ─── EVENT → SECTOR MAPPING ──────────────────────────────────────────────────
EVENT_SECTORS = {
    "Non-Farm Payrolls":         ["Technology","Consumer Discretionary","Financials","Communication"],
    "Nonfarm Payrolls":          ["Technology","Consumer Discretionary","Financials"],
    "Unemployment":              ["Technology","Consumer Discretionary","Financials"],
    "CPI":                       ["Technology","Communication","Consumer Discretionary"],
    "Inflation":                 ["Technology","Communication","Consumer Discretionary"],
    "PPI":                       ["Technology","Consumer Discretionary","Industrials"],
    "PCE":                       ["Technology","Communication","Consumer Discretionary"],
    "GDP":                       ["Technology","Consumer Discretionary","Industrials","Materials"],
    "Retail Sales":              ["Consumer Discretionary","Technology","Financials"],
    "ISM Manufacturing":         ["Industrials","Materials","Technology"],
    "ISM Services":              ["Financials","Consumer Discretionary","Technology"],
    "PMI":                       ["Technology","Industrials","Materials"],
    "Consumer Confidence":       ["Consumer Discretionary","Financials","Technology"],
    "Consumer Sentiment":        ["Consumer Discretionary","Financials"],
    "Jobless Claims":            ["Technology","Consumer Discretionary","Financials"],
    "Initial Claims":            ["Technology","Consumer Discretionary"],
    "Durable Goods":             ["Industrials","Technology"],
    "Housing Starts":            ["Materials","Industrials","Real Estate"],
    "Existing Home Sales":       ["Real Estate","Financials","Materials"],
    "New Home Sales":            ["Real Estate","Materials","Industrials"],
    "Trade Balance":             ["Technology","Industrials","Financials"],
    "FOMC":                      ["Technology","Communication","Consumer Discretionary"],
    "Fed":                       ["Technology","Communication","Financials"],
    "Crude Oil":                 ["Energy"],
    "Oil":                       ["Energy"],
    "ADP":                       ["Technology","Financials","Consumer Discretionary"],
}

# Events where LOWER actual vs forecast = bullish for stocks
INVERSE_EVENTS = {
    "cpi","inflation","ppi","pce","unemployment","jobless claims",
    "initial claims","crude oil inventories","inventory"
}

# ─── SCRAPER: TRADING ECONOMICS ──────────────────────────────────────────────
def scrape_calendar():
    """Fetch today's USD economic events from Trading Economics."""
    events = []
    try:
        url  = "https://tradingeconomics.com/united-states/calendar"
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        rows = soup.select("tr[data-url], tr.calendar-item, table#calendar tbody tr")
        if not rows:
            rows = soup.select("tr")

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            text_vals = [c.get_text(strip=True) for c in cells]

            # Skip rows that don't look like event rows
            if not any(text_vals):
                continue

            # Try to extract fields by position or data attributes
            event_name = (row.get("data-event") or
                          row.get("data-symbol") or
                          (cells[2].get_text(strip=True) if len(cells) > 2 else ""))

            if not event_name or len(event_name) < 3:
                continue

            # Time (first cell usually)
            time_str = cells[0].get_text(strip=True) if cells else ""

            # Impact — look for color classes or data attrs
            impact = "Medium"
            imp_el = row.find(class_=lambda c: c and any(
                x in c.lower() for x in ["high","medium","low","red","orange","yellow"]))
            if imp_el:
                cls_str = " ".join(imp_el.get("class", []))
                if any(x in cls_str.lower() for x in ["high","red","danger"]):
                    impact = "High"
                elif any(x in cls_str.lower() for x in ["medium","orange","warning"]):
                    impact = "Medium"
                else:
                    impact = "Low"

            # Actual / Forecast / Previous — last 3 numeric-looking cells
            actual   = row.get("data-actual","")   or (cells[-3].get_text(strip=True) if len(cells) >= 3 else "")
            forecast = row.get("data-forecast","") or (cells[-2].get_text(strip=True) if len(cells) >= 2 else "")
            previous = row.get("data-previous","") or (cells[-1].get_text(strip=True) if len(cells) >= 1 else "")

            events.append({
                "time":     time_str,
                "event":    event_name,
                "impact":   impact,
                "actual":   actual,
                "forecast": forecast,
                "previous": previous,
            })

        print(f"    Trading Economics: {len(events)} events parsed")
    except Exception as e:
        print(f"    Scrape failed ({e}), using news-only mode")

    return events


def scrape_calendar_fallback():
    """Backup: fetch generic economic news from Yahoo Finance."""
    headlines = []
    for sym in ["^GSPC", "^NDX", "SPY", "QQQ", "^VIX", "TLT", "GLD"]:
        try:
            for n in (yf.Ticker(sym).news or [])[:3]:
                t = n.get("content", {}).get("title") or n.get("title", "")
                if t and t not in headlines:
                    headlines.append(t)
        except:
            pass
    return headlines[:10]


# ─── SENTIMENT ANALYSIS ──────────────────────────────────────────────────────
def parse_num(s):
    if not s or s in ("-", "—", ""):
        return None
    try:
        cleaned = s.strip().replace(",","").replace("%","")
        cleaned = cleaned.replace("K","e3").replace("M","e6").replace("B","e9")
        return float(cleaned)
    except:
        return None


def event_direction(ev):
    name     = ev["event"].lower()
    actual   = parse_num(ev["actual"])
    forecast = parse_num(ev["forecast"])
    if actual is None or forecast is None:
        return "pending"

    inverse = any(kw in name for kw in INVERSE_EVENTS)
    beat    = (actual < forecast) if inverse else (actual > forecast)
    diff    = abs(actual - forecast) / (abs(forecast) + 1e-9) * 100

    if diff < 0.3:
        return "neutral"
    return "bullish" if beat else "bearish"


def get_sentiment(events):
    weight  = {"High": 3, "Medium": 1, "Low": 0}
    bull, bear = 0, 0
    bullish_sectors = set()
    analysed = []

    for ev in events:
        d = event_direction(ev)
        w = weight.get(ev["impact"], 1)
        if d == "bullish":
            bull += w
            for kw, sects in EVENT_SECTORS.items():
                if kw.lower() in ev["event"].lower():
                    bullish_sectors.update(sects)
                    break
        elif d == "bearish":
            bear += w
        analysed.append({**ev, "direction": d})

    net = bull - bear
    if net >= 5:     label = "Strong Bullish  &#128640;"
    elif net >= 2:   label = "Bullish  &#128994;"
    elif net >= 0:   label = "Mildly Bullish  &#128308;"
    elif net >= -2:  label = "Mixed / Cautious  &#128310;"
    else:            label = "Bearish  &#128308;"

    pending = not any(ev["actual"] for ev in events)
    if pending:
        label = "Pre-Events — Watch Releases  &#9203;"

    if not bullish_sectors:
        bullish_sectors = {"Technology", "Consumer Discretionary", "Communication",
                           "Semiconductors", "Mega Cap Tech"}

    return label, net, list(bullish_sectors), analysed


# ─── STOCK SELECTION ─────────────────────────────────────────────────────────
def score_stocks(tickers):
    """Score stocks by pre-market move + volume + news. Long-only filter."""
    if not tickers:
        return []

    scored = []
    chunk  = 40
    all_data = {}

    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i+chunk]
        try:
            raw = yf.download(batch, period="5d", interval="1d",
                              auto_adjust=True, progress=False,
                              group_by="ticker", threads=True)
            for tk in batch:
                try:
                    df = raw[tk] if len(batch) > 1 else raw
                    all_data[tk] = df.dropna() if df is not None else None
                except:
                    all_data[tk] = None
        except:
            pass

    for tk in tickers:
        try:
            df = all_data.get(tk)
            price = day_chg = vol_ratio = 0.0

            if df is not None and len(df) >= 2:
                price    = float(df["Close"].iloc[-1])
                prev     = float(df["Close"].iloc[-2])
                day_chg  = (price - prev) / prev * 100
                vol      = float(df["Volume"].iloc[-1])
                avg_vol  = float(df["Volume"].mean()) + 1
                vol_ratio = vol / avg_vol

            # Pre-market quote
            pre_chg = 0.0
            try:
                fi = yf.Ticker(tk).fast_info
                if hasattr(fi, "last_price") and hasattr(fi, "previous_close") and fi.previous_close:
                    pre_chg = (fi.last_price - fi.previous_close) / fi.previous_close * 100
            except:
                pass

            # Long-only filter: skip stocks with pre-market gap down > 1%
            if pre_chg < -1.0 and day_chg < -1.0:
                continue

            # News
            headlines = []
            news_score = 0
            try:
                for n in (yf.Ticker(tk).news or [])[:3]:
                    t = n.get("content", {}).get("title") or n.get("title", "")
                    if t:
                        headlines.append(t)
                        hl = t.lower()
                        pos = ["beat","surge","rise","rally","upgrade","buy","strong",
                               "record","growth","profit","expands","wins","soars"]
                        neg = ["miss","fall","drop","downgrade","sell","loss","weak",
                               "cuts","warning","concern","halt","suspend","fraud"]
                        if any(w in hl for w in pos): news_score += 4
                        if any(w in hl for w in neg): news_score -= 5
            except:
                pass

            # Skip if heavy negative news
            if news_score <= -5:
                continue

            score = (pre_chg * 3) + (day_chg * 1.5) + ((vol_ratio - 1) * 3) + news_score

            scored.append({
                "ticker":    tk,
                "score":     round(score, 2),
                "price":     round(price, 2),
                "pre_chg":   round(pre_chg, 2),
                "day_chg":   round(day_chg, 2),
                "vol_ratio": round(vol_ratio, 2),
                "news":      headlines,
            })
        except:
            pass

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def pick_stocks(bullish_sectors):
    """Build candidate lists from bullish sectors and score them."""
    sp500_cands, ndx_cands = [], []

    for sector in bullish_sectors:
        sp500_cands.extend(SP500_SECTORS.get(sector, []))
        ndx_cands.extend(NDX_SECTORS.get(sector, []))

    # Always include core tech + mega caps
    for sec in ["Technology", "Consumer Discretionary", "Communication"]:
        sp500_cands.extend(SP500_SECTORS.get(sec, []))
    for sec in ["Mega Cap Tech", "Semiconductors", "Software / Cloud"]:
        ndx_cands.extend(NDX_SECTORS.get(sec, []))

    sp500_cands = list(dict.fromkeys(sp500_cands))[:100]
    ndx_cands   = list(dict.fromkeys(ndx_cands))[:80]

    print(f"    Scoring {len(sp500_cands)} S&P 500 candidates...")
    sp500_scored = score_stocks(sp500_cands)

    print(f"    Scoring {len(ndx_cands)} NDX candidates...")
    ndx_scored = score_stocks(ndx_cands)

    return sp500_scored[:TOP_N], ndx_scored[:TOP_N]


# ─── REPORT BUILDER ──────────────────────────────────────────────────────────
def index_snapshot():
    snap = {}
    for label, sym in [("S&P 500","^GSPC"),("NDX 100","^NDX"),("VIX","^VIX"),("Dow","^DJI")]:
        try:
            fi = yf.Ticker(sym).fast_info
            snap[label] = {
                "price": round(fi.last_price, 2),
                "chg":   round((fi.last_price - fi.previous_close) / fi.previous_close * 100, 2),
            }
        except:
            snap[label] = None
    return snap


def build_html(events, sentiment, net, bullish_sectors, analysed,
               sp500_picks, ndx_picks, mkt_news, snap):

    today = datetime.now().strftime("%A, %B %d, %Y")
    time_ = datetime.now().strftime("%I:%M %p ET")

    sent_col = ("#3fb950" if "Bullish" in sentiment or "Mildly" in sentiment
                else "#f0883e" if "Mixed" in sentiment or "Pre-Events" in sentiment
                else "#f85149")

    # ── Snapshot ──
    snap_cells = ""
    for lbl, d in snap.items():
        if not d: continue
        col = "#3fb950" if d["chg"] >= 0 else "#f85149"
        arrow = "▲" if d["chg"] >= 0 else "▼"
        snap_cells += (f"<td style='padding:4px 20px 4px 0'><b>{lbl}</b></td>"
                       f"<td style='padding:4px 16px 4px 0'>{d['price']:,.2f}</td>"
                       f"<td style='color:{col};padding:4px 20px 4px 0'>{arrow} {d['chg']:+.2f}%</td>")

    # ── Calendar rows ──
    impact_col = {"High":"#da3633","Medium":"#f0883e","Low":"#388bfd"}

    def dir_badge(d):
        if d == "bullish":  return "<span style='color:#3fb950'>▲ Bullish</span>"
        if d == "bearish":  return "<span style='color:#f85149'>▼ Bearish</span>"
        if d == "pending":  return "<span style='color:#8b949e'>⏳ Pending</span>"
        return "<span style='color:#8b949e'>— Neutral</span>"

    cal_rows = ""
    shown = sorted(analysed, key=lambda x: x["impact"]=="High", reverse=True)[:15]
    for ev in shown:
        ic = impact_col.get(ev["impact"],"#555")
        cal_rows += f"""<tr>
          <td style='color:#8b949e;white-space:nowrap'>{ev['time']}</td>
          <td style='font-weight:500'>{ev['event']}</td>
          <td><span style='background:{ic};color:#fff;padding:1px 7px;border-radius:10px;font-size:0.75em'>{ev['impact']}</span></td>
          <td style='font-weight:bold;color:#e6edf3'>{ev['actual'] or '<span style=\"color:#8b949e\">Pending</span>'}</td>
          <td style='color:#8b949e'>{ev['forecast'] or '—'}</td>
          <td style='color:#8b949e'>{ev['previous'] or '—'}</td>
          <td>{dir_badge(ev['direction'])}</td>
        </tr>"""

    no_cal = not cal_rows
    if no_cal:
        cal_rows = "<tr><td colspan='7' style='text-align:center;color:#8b949e;padding:14px'>No USD events today — news-driven selection mode</td></tr>"

    # ── Sector badges ──
    sector_badges = "".join(
        f"<span style='background:#1f6feb;color:#fff;padding:2px 9px;border-radius:12px;"
        f"font-size:0.78em;margin:2px;display:inline-block'>{s}</span>"
        for s in bullish_sectors[:8])

    # ── News headlines ──
    news_li = "".join(f"<li>{n}</li>" for n in mkt_news[:8]) or "<li>No headlines available</li>"

    # ── Pending warning ──
    pending_banner = ""
    if any(ev["direction"] == "pending" for ev in analysed):
        pending_banner = """<div style='background:#162032;border-left:3px solid #388bfd;
          padding:10px 16px;margin:10px 0;border-radius:4px;font-size:0.9em'>
          <b>⚠ High-impact events releasing today.</b>
          Stocks below are pre-positioned assuming a bullish release.
          If data disappoints, skip or wait 5–10 min after release before entering.
        </div>"""

    # ── Stock cards ──
    def make_cards(picks, title):
        if not picks:
            return f"<h2>{title}</h2><p style='color:#8b949e'>No strong long candidates today.</p>"

        cards = ""
        for i, p in enumerate(picks):
            rank   = i + 1
            border = "#3fb950" if rank <= 3 else "#30363d"
            pc_col = "#3fb950" if p["pre_chg"] >= 0 else "#f85149"
            dc_col = "#3fb950" if p["day_chg"] >= 0 else "#f85149"

            reasons = []
            if abs(p["pre_chg"]) > 0.2:
                reasons.append(f"Pre-mkt {'▲' if p['pre_chg']>0 else '▼'} {p['pre_chg']:+.2f}%")
            if p["vol_ratio"] > 1.4:
                reasons.append(f"Volume {p['vol_ratio']:.1f}× avg")
            if p["news"]:
                reasons.append("News catalyst")
            if not reasons:
                reasons.append("Sector macro tailwind")
            why = "  ·  ".join(reasons)

            news_html = ""
            for n in p["news"][:2]:
                news_html += (f"<div style='font-size:0.78em;color:#8b949e;border-left:2px solid #30363d;"
                              f"padding-left:8px;margin:3px 0'>▸ {n[:115]}</div>")

            cards += f"""
<div style='background:#161b22;border:1px solid {border};border-radius:8px;padding:15px;'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>
    <span style='font-size:1.3em;font-weight:bold;color:#3fb950'>#{rank} {p['ticker']}</span>
    <span style='font-size:0.88em'>Pre-mkt <b style='color:{pc_col}'>{p['pre_chg']:+.2f}%</b></span>
  </div>
  <div style='color:#8b949e;font-size:0.83em;margin-bottom:6px'>
    ${p['price']} &nbsp;|&nbsp; Yesterday <span style='color:{dc_col}'>{p['day_chg']:+.2f}%</span>
    &nbsp;|&nbsp; Vol {p['vol_ratio']:.1f}×
  </div>
  <div style='font-size:0.83em;color:#adbac7;margin-bottom:5px'>📈 {why}</div>
  <div style='font-size:0.82em;color:#3fb950;margin-bottom:6px'>
    → Wait for EMA 9 cross above EMA 21 on 5m or 10m · Trailing stop 1.5%
  </div>
  {news_html}
</div>"""

        return (f"<h2>{title}</h2>"
                f"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px'>"
                f"{cards}</div>")

    sp500_html = make_cards(sp500_picks, f"S&amp;P 500 — Top {TOP_N} Long Candidates")
    ndx_html   = make_cards(ndx_picks,   f"Nasdaq 100 — Top {TOP_N} Long Candidates")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Market Report — {today}</title>
<style>
  *   {{ box-sizing:border-box; margin:0; padding:0 }}
  body{{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:#0d1117;color:#e6edf3;padding:24px;line-height:1.6 }}
  h1  {{ color:#58a6ff;font-size:1.5em;border-bottom:1px solid #30363d;
         padding-bottom:10px;margin-bottom:14px }}
  h2  {{ color:#79c0ff;font-size:1.1em;margin:28px 0 12px }}
  table {{ width:100%;border-collapse:collapse;font-size:0.87em }}
  th  {{ background:#161b22;color:#8b949e;padding:7px 10px;text-align:left;
         border-bottom:1px solid #30363d }}
  td  {{ padding:7px 10px;border-bottom:1px solid #21262d;vertical-align:middle }}
  .box{{ background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 18px }}
  .ctx{{ background:#161b22;border-left:3px solid #388bfd;
         padding:12px 16px;border-radius:4px;margin-bottom:12px }}
  .risk{{background:#161b22;border-left:3px solid #f0883e;
          padding:12px 16px;border-radius:4px }}
  ul  {{ padding-left:20px;color:#adbac7;font-size:0.9em }}
  li  {{ margin:4px 0 }}
  footer{{ color:#6e7681;font-size:0.75em;margin-top:40px;
            border-top:1px solid #21262d;padding-top:12px }}
</style>
</head>
<body>

<h1>🌅 Morning Market Report — {today}</h1>
<p style='color:#8b949e;font-size:0.85em;margin-bottom:20px'>
  {time_} &nbsp;|&nbsp; Source: Trading Economics + Yahoo Finance
  &nbsp;|&nbsp; Strategy: EMA 9/21 · 5m / 10m · Long only · 1.5% trail stop
</p>

<h2>Index Snapshot</h2>
<div class='box' style='display:inline-block;margin-bottom:8px'>
  <table style='width:auto'><tr>{snap_cells}</tr></table>
</div>

<h2>US Economic Calendar</h2>
{pending_banner}
<div class='box' style='padding:4px'>
  <table>
    <tr>
      <th>Time (ET)</th><th>Event</th><th>Impact</th>
      <th>Actual</th><th>Forecast</th><th>Previous</th><th>Signal</th>
    </tr>
    {cal_rows}
  </table>
</div>

<h2>Overall Market Sentiment</h2>
<div class='ctx'>
  <div style='font-size:1.25em;font-weight:bold;color:{sent_col}'>{sentiment}</div>
  <div style='color:#8b949e;font-size:0.88em;margin:6px 0'>
    Macro score: {net:+d} &nbsp;|&nbsp; Favoured sectors:
  </div>
  <div style='margin-top:6px'>{sector_badges}</div>
</div>

<h2>Market Headlines</h2>
<div class='ctx'><ul>{news_li}</ul></div>

{sp500_html}

{ndx_html}

<h2>Trading Notes</h2>
<div class='risk'>
  <ul>
    <li><b>Long only</b> — only enter when EMA 9 crosses above EMA 21 on the 5m or 10m chart</li>
    <li>Let the <b>first 5-minute bar close</b> after market open before touching anything</li>
    <li><b>1.5% trailing stop</b> on every trade — no exceptions, no manual overrides</li>
    <li>If a ⬛ High-impact event releases during the session — wait 5–10 min after the number drops before entering</li>
    <li>Pre-market movers with volume 1.5× or higher tend to continue the first 30–60 min — prioritise those</li>
    <li>VIX &gt; 20 = choppier price action → tighten position size. VIX &lt; 15 = cleaner trends → normal size</li>
  </ul>
</div>

<footer>
  Generated by Claude Code &nbsp;·&nbsp; Data: Trading Economics + Yahoo Finance
  &nbsp;·&nbsp; Not financial advice.
</footer>
</body>
</html>"""


# ─── EMAIL ───────────────────────────────────────────────────────────────────
def send_email(html, subject):
    sender    = os.environ.get("GMAIL_USER", "")
    password  = os.environ.get("GMAIL_APP_PASSWORD", "")
    recipient = os.environ.get("REPORT_EMAIL", "")
    if not all([sender, password, recipient]):
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Market Report <{sender}>"
    msg["To"]      = recipient
    msg.attach(MIMEText(html, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
    return True


# ─── DST CHECK ───────────────────────────────────────────────────────────────
def check_et_hour():
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.hour != 8 or now_et.minute < 28:
            print(f"Not 8:30 AM ET ({now_et.strftime('%I:%M %p ET')}), skipping.")
            sys.exit(0)
    except:
        pass


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    if CLOUD_MODE:
        check_et_hour()

    print(f"\n{'='*62}")
    print(f"  MACRO REPORT  {datetime.now().strftime('%A %B %d, %Y  %I:%M %p')}")
    print(f"  Mode: {'CLOUD (email)' if CLOUD_MODE else 'LOCAL (browser)'}")
    print(f"{'='*62}")

    print("\n  [1/6] Index snapshot...")
    snap = index_snapshot()
    for l, d in snap.items():
        if d: print(f"    {l}: {d['price']:>10,.2f}  ({d['chg']:+.2f}%)")

    print("\n  [2/6] Scraping economic calendar...")
    events = scrape_calendar()
    if not events:
        events = []    # graceful empty

    print("\n  [3/6] Analyzing macro sentiment...")
    sentiment, net, bullish_sectors, analysed = get_sentiment(events)
    print(f"    {sentiment}  (score {net:+d})")
    print(f"    Bullish sectors: {', '.join(bullish_sectors[:5])}")

    print("\n  [4/6] Fetching market headlines...")
    mkt_news = scrape_calendar_fallback()
    print(f"    {len(mkt_news)} headlines")

    print("\n  [5/6] Selecting stocks...")
    sp500_picks, ndx_picks = pick_stocks(bullish_sectors)
    print(f"    S&P 500: {', '.join(p['ticker'] for p in sp500_picks)}")
    print(f"    NDX:     {', '.join(p['ticker'] for p in ndx_picks)}")

    print("\n  [6/6] Building report...")
    html    = build_html(events, sentiment, net, bullish_sectors, analysed,
                         sp500_picks, ndx_picks, mkt_news, snap)
    subject = (f"Market Report {datetime.now().strftime('%a %b %d')} "
               f"| {sentiment.split()[0]} "
               f"| {' '.join(p['ticker'] for p in sp500_picks[:3])}...")

    if CLOUD_MODE:
        send_email(html, subject)
        print(f"    ✓ Email sent → {os.environ['REPORT_EMAIL']}")
    else:
        import webbrowser
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fname = os.path.join(OUTPUT_DIR, f"macro_{datetime.now().strftime('%Y%m%d')}.html")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"    Saved: {fname}")
        webbrowser.open(f"file:///{fname}")

    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    main()
