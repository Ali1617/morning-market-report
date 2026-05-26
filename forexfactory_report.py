#!/usr/bin/env python3
"""
Macro-Driven Morning Stock Report  —  Premium Edition
- Scrapes Trading Economics for USD economic events
- Scores S&P 500 + NDX 100 stocks by macro tailwind + momentum + news
- Publishes a full premium HTML report to GitHub Pages (docs/index.html)
- Emails a clean summary card with a single "Open Report" link
"""

import os, sys, smtplib, warnings, json
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

warnings.filterwarnings("ignore")

CLOUD_MODE  = bool(os.environ.get("GMAIL_USER"))
REPORT_URL  = os.environ.get("REPORT_URL", "https://ali1617.github.io/morning-market-report/")
OUTPUT_DIR  = "docs"           # GitHub Pages serves from /docs
TOP_N       = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─── STOCK UNIVERSE ───────────────────────────────────────────────────────────
SP500_SECTORS = {
    "Technology":             ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","CSCO","TXN","QCOM",
                               "AMD","MU","ADI","AMAT","KLAC","LRCX","IBM","INTC","HPQ","DELL"],
    "Consumer Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","MAR",
                               "HLT","RCL","CCL","F","GM"],
    "Financials":             ["JPM","BAC","WFC","GS","MS","BLK","AXP","V","MA","SPGI",
                               "MCO","C","USB","CME","SCHW","COF"],
    "Healthcare":             ["UNH","LLY","JNJ","ABBV","MRK","TMO","ABT","DHR","BSX",
                               "ISRG","ELV","VRTX","REGN","CI","HUM"],
    "Industrials":            ["GE","CAT","RTX","HON","DE","BA","UNP","UPS","FDX","ETN",
                               "ITW","NOC","LMT","NSC","CSX","MMM"],
    "Energy":                 ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","BKR","HAL","OXY"],
    "Communication":          ["META","GOOGL","NFLX","DIS","CMCSA","T","VZ","CHTR","EA"],
    "Materials":              ["LIN","APD","SHW","FCX","NEM","ECL","ALB","NUE"],
    "Consumer Staples":       ["WMT","PG","COST","KO","PEP","PM","MDLZ","CL"],
}

NDX_SECTORS = {
    "Semiconductors":         ["NVDA","AMD","AVGO","INTC","QCOM","TXN","MU","AMAT","KLAC",
                               "LRCX","SNPS","CDNS","MRVL","ON","MCHP","GFS","NXPI","SMCI"],
    "Mega Cap Tech":          ["AAPL","MSFT","GOOGL","AMZN","META","TSLA"],
    "Software / Cloud":       ["ADBE","INTU","CRWD","PANW","FTNT","ZS","DDOG","PLTR",
                               "WDAY","TEAM","ADSK","TTD","COIN","HOOD"],
    "Consumer / Retail":      ["COST","SBUX","MELI","ABNB","NFLX","BKNG"],
    "Biotech / Health":       ["GILD","AMGN","VRTX","REGN","MRNA","BIIB","IDXX"],
    "Industrials / Energy":   ["PCAR","HON","BKR","FSLR","CEG","CSX","FAST"],
    "Other":                  ["MNST","KDP","MDLZ","PEP","CHTR","CMCSA","EA"],
}

EVENT_SECTORS = {
    "Non-Farm Payrolls":      ["Technology","Consumer Discretionary","Financials"],
    "Nonfarm Payrolls":       ["Technology","Consumer Discretionary","Financials"],
    "Unemployment":           ["Technology","Consumer Discretionary","Financials"],
    "CPI":                    ["Technology","Communication","Consumer Discretionary"],
    "Inflation":              ["Technology","Communication","Consumer Discretionary"],
    "PPI":                    ["Technology","Consumer Discretionary","Industrials"],
    "PCE":                    ["Technology","Communication","Consumer Discretionary"],
    "GDP":                    ["Technology","Consumer Discretionary","Industrials","Materials"],
    "Retail Sales":           ["Consumer Discretionary","Technology","Financials"],
    "ISM Manufacturing":      ["Industrials","Materials","Technology"],
    "ISM Services":           ["Financials","Consumer Discretionary","Technology"],
    "PMI":                    ["Technology","Industrials","Materials"],
    "Consumer Confidence":    ["Consumer Discretionary","Financials","Technology"],
    "Consumer Sentiment":     ["Consumer Discretionary","Financials"],
    "Jobless Claims":         ["Technology","Consumer Discretionary","Financials"],
    "Initial Claims":         ["Technology","Consumer Discretionary"],
    "Durable Goods":          ["Industrials","Technology"],
    "Housing Starts":         ["Materials","Industrials"],
    "Trade Balance":          ["Technology","Industrials"],
    "FOMC":                   ["Technology","Communication","Consumer Discretionary"],
    "Fed":                    ["Technology","Communication","Financials"],
    "Crude Oil":              ["Energy"],
    "ADP":                    ["Technology","Financials","Consumer Discretionary"],
}

INVERSE_EVENTS = {"cpi","inflation","ppi","pce","unemployment","jobless claims",
                  "initial claims","crude oil inventories","inventory"}

# ─── DATA FETCHING ────────────────────────────────────────────────────────────
def scrape_calendar():
    try:
        resp = requests.get("https://tradingeconomics.com/united-states/calendar",
                            headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        events, seen = [], set()
        for row in soup.select("tr"):
            cells = row.find_all("td")
            if len(cells) < 3: continue
            name = (row.get("data-event") or row.get("data-symbol") or
                    cells[2].get_text(strip=True) if len(cells) > 2 else "")
            if not name or len(name) < 3 or name in seen: continue
            seen.add(name)
            impact = "Medium"
            for el in row.find_all(True):
                cls = " ".join(el.get("class", []))
                if any(x in cls.lower() for x in ["high","red","danger"]):
                    impact = "High"; break
                elif any(x in cls.lower() for x in ["medium","orange","warning"]):
                    impact = "Medium"
            events.append({
                "time":     cells[0].get_text(strip=True) if cells else "",
                "event":    name,
                "impact":   impact,
                "actual":   row.get("data-actual","")   or (cells[-3].get_text(strip=True) if len(cells)>=3 else ""),
                "forecast": row.get("data-forecast","") or (cells[-2].get_text(strip=True) if len(cells)>=2 else ""),
                "previous": row.get("data-previous","") or (cells[-1].get_text(strip=True) if len(cells)>=1 else ""),
            })
        return events
    except Exception as e:
        print(f"    Calendar scrape failed: {e}")
        return []

def parse_num(s):
    if not s or s in ("-","—",""): return None
    try:
        return float(s.strip().replace(",","").replace("%","")
                     .replace("K","e3").replace("M","e6").replace("B","e9"))
    except: return None

def event_direction(ev):
    name = ev["event"].lower()
    a, f = parse_num(ev["actual"]), parse_num(ev["forecast"])
    if a is None or f is None: return "pending"
    inverse = any(kw in name for kw in INVERSE_EVENTS)
    beat    = (a < f) if inverse else (a > f)
    if abs(a - f) / (abs(f) + 1e-9) * 100 < 0.3: return "neutral"
    return "bullish" if beat else "bearish"

def get_sentiment(events):
    w = {"High":3,"Medium":1,"Low":0}
    bull = bear = 0
    sectors = set()
    analysed = []
    for ev in events:
        d = event_direction(ev)
        wt = w.get(ev["impact"],1)
        if d == "bullish":
            bull += wt
            for kw, s in EVENT_SECTORS.items():
                if kw.lower() in ev["event"].lower():
                    sectors.update(s); break
        elif d == "bearish":
            bear += wt
        analysed.append({**ev, "direction": d})
    net = bull - bear
    if net >= 5:   label, emoji = "Strong Bullish", "🚀"
    elif net >= 2: label, emoji = "Bullish",        "📈"
    elif net >= 0: label, emoji = "Mildly Bullish", "🟡"
    elif net >=-2: label, emoji = "Mixed",          "⚠️"
    else:          label, emoji = "Bearish",        "📉"
    if not any(ev["actual"] for ev in events):
        label, emoji = "Pre-Events", "⏳"
    if not sectors:
        sectors = {"Technology","Consumer Discretionary","Communication",
                   "Semiconductors","Mega Cap Tech"}
    return label, emoji, net, list(sectors), analysed

def get_market_news():
    items = []
    for sym in ["^GSPC","^NDX","SPY","QQQ","^VIX"]:
        try:
            for n in (yf.Ticker(sym).news or [])[:3]:
                t = n.get("content",{}).get("title") or n.get("title","")
                if t and t not in items: items.append(t)
        except: pass
    return items[:8]

def index_snapshot():
    snap = {}
    for label, sym in [("S&P 500","^GSPC"),("Nasdaq 100","^NDX"),("VIX","^VIX"),("Dow Jones","^DJI")]:
        try:
            fi = yf.Ticker(sym).fast_info
            snap[label] = {"price": round(fi.last_price,2),
                           "chg":   round((fi.last_price-fi.previous_close)/fi.previous_close*100,2)}
        except: snap[label] = None
    return snap

def _calc_rsi(close_arr, period=14):
    """Wilder RSI from a numpy/list of closes."""
    s = pd.Series(close_arr, dtype=float)
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def score_stocks(tickers, ref_day_chg=0.0):
    """
    Multi-factor long candidate scorer.

    Hard filters (eliminate outright):
      • Price < $8              — poor intraday spreads
      • Day change < -2.5%     — broken momentum, avoid
      • Pre-mkt < -1.5% AND day < -1.0% — confirmed sell-off
      • RSI > 80               — overbought, no room to run
      • RSI < 35               — in a downtrend
      • News score <= -5       — negative catalyst

    Scoring (higher = better long):
      pre_chg  × 3.5   — pre-market momentum (strongest signal)
      day_chg  × 2.0   — yesterday's close direction
      (vol_ratio-1)×2.5 — volume surge vs 30-day average
      rel_str  × 1.5   — outperforming SPY/QQQ today
      rsi_bonus         — +2.5 if RSI 45-68 (sweet spot), +1 if 68-78
      ema_bonus         — +2 if price > 20-day EMA, -1 if below
      gap_bonus         — +2 if pre-mkt gap >0.5%, +0.8 if >0.2%
      vol_surge_bonus   — extra +1.5 if volume > 2× average
      news_score        — ±points from headline keywords
    """
    if not tickers:
        return []

    # ── Batch-download 30 days of daily data (needed for RSI14 + EMA20) ──────
    all_data = {}
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i+40]
        try:
            raw = yf.download(batch, period="30d", interval="1d",
                              auto_adjust=True, progress=False,
                              group_by="ticker", threads=True)
            for tk in batch:
                try:
                    all_data[tk] = (raw[tk] if len(batch) > 1 else raw).dropna()
                except:
                    all_data[tk] = None
        except:
            pass

    POS_WORDS = {"beat","surge","rise","rally","upgrade","buy","strong","record",
                 "growth","profit","raises","exceed","above","jumps","soars","wins"}
    NEG_WORDS = {"miss","fall","drop","downgrade","sell","loss","weak","cuts",
                 "warning","fraud","below","recall","investigation","probe","halt"}

    scored = []
    for tk in tickers:
        try:
            df = all_data.get(tk)
            if df is None or len(df) < 5:
                continue

            close_arr = df["Close"].values.astype(float)
            price     = close_arr[-1]

            # ── Hard filters ───────────────────────────────────────────────
            if price < 8:                          # penny / micro-cap
                continue

            prev    = close_arr[-2]
            day_chg = (price - prev) / prev * 100

            if day_chg < -2.5:                     # hard sell-off yesterday
                continue

            # ── Volume ────────────────────────────────────────────────────
            avg_vol   = float(df["Volume"].mean()) + 1
            vol_ratio = float(df["Volume"].iloc[-1]) / avg_vol

            # ── Pre-market ────────────────────────────────────────────────
            pre_chg = 0.0
            try:
                fi = yf.Ticker(tk).fast_info
                if hasattr(fi, "last_price") and fi.previous_close:
                    pre_chg = (fi.last_price - fi.previous_close) / fi.previous_close * 100
            except:
                pass

            if pre_chg < -1.5 and day_chg < -1.0: # confirmed sell-off
                continue

            # ── RSI(14) ───────────────────────────────────────────────────
            rsi = _calc_rsi(close_arr) if len(close_arr) >= 15 else 50.0
            if rsi > 80 or rsi < 35:               # overbought / downtrend
                continue

            # ── 20-day EMA trend filter ───────────────────────────────────
            ema20       = float(pd.Series(close_arr).ewm(span=20, adjust=False).mean().iloc[-1])
            above_ema20 = price > ema20

            # ── Relative strength vs market ───────────────────────────────
            rel_strength = day_chg - ref_day_chg   # positive = outperforming index

            # ── News ──────────────────────────────────────────────────────
            headlines, news_score = [], 0
            try:
                for n in (yf.Ticker(tk).news or [])[:3]:
                    t = n.get("content", {}).get("title") or n.get("title", "")
                    if t:
                        headlines.append(t)
                        words = set(t.lower().split())
                        if words & POS_WORDS: news_score += 4
                        if words & NEG_WORDS: news_score -= 5
            except:
                pass

            if news_score <= -5:
                continue

            # ── Composite score ───────────────────────────────────────────
            rsi_bonus      = 2.5 if 45 <= rsi <= 68 else (1.0 if rsi <= 78 else 0.0)
            ema_bonus      = 2.0 if above_ema20 else -1.0
            gap_bonus      = 2.0 if pre_chg > 0.5 else (0.8 if pre_chg > 0.2 else 0.0)
            vol_surge_bonus= 1.5 if vol_ratio > 2.0 else 0.0

            score = (
                pre_chg * 3.5
                + day_chg * 2.0
                + (vol_ratio - 1) * 2.5
                + rel_strength * 1.5
                + rsi_bonus
                + ema_bonus
                + gap_bonus
                + vol_surge_bonus
                + news_score
            )

            scored.append({
                "ticker":      tk,
                "score":       round(score, 2),
                "price":       round(price, 2),
                "pre_chg":     round(pre_chg, 2),
                "day_chg":     round(day_chg, 2),
                "vol_ratio":   round(vol_ratio, 2),
                "rsi":         round(rsi, 1),
                "above_ema20": above_ema20,
                "rel_str":     round(rel_strength, 2),
                "news":        headlines,
            })
        except:
            pass

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def pick_stocks(sectors):
    # ── Reference return: SPY for S&P, QQQ for NDX ───────────────────────────
    spy_chg = qcq_chg = 0.0
    try:
        spy_fi  = yf.Ticker("SPY").fast_info
        spy_chg = (spy_fi.last_price - spy_fi.previous_close) / spy_fi.previous_close * 100
    except: pass
    try:
        qqq_fi  = yf.Ticker("QQQ").fast_info
        qcq_chg = (qqq_fi.last_price - qqq_fi.previous_close) / qqq_fi.previous_close * 100
    except: pass

    # ── Candidate pool ────────────────────────────────────────────────────────
    sp_c, nd_c = [], []

    # Priority: macro-favoured sectors first
    for s in sectors:
        sp_c.extend(SP500_SECTORS.get(s, []))
        nd_c.extend(NDX_SECTORS.get(s, []))

    # Always include core high-liquidity sectors as a baseline
    for s in list(SP500_SECTORS.keys()):
        sp_c.extend(SP500_SECTORS.get(s, []))
    for s in list(NDX_SECTORS.keys()):
        nd_c.extend(NDX_SECTORS.get(s, []))

    sp_c = list(dict.fromkeys(sp_c))[:120]
    nd_c = list(dict.fromkeys(nd_c))[:100]

    print(f"    Scoring {len(sp_c)} S&P candidates...")
    print(f"    Scoring {len(nd_c)} NDX candidates...")
    return score_stocks(sp_c, ref_day_chg=spy_chg)[:TOP_N], \
           score_stocks(nd_c, ref_day_chg=qcq_chg)[:TOP_N]

# ─── FULL REPORT HTML (GitHub Pages) ─────────────────────────────────────────
def build_full_report(events, sentiment, emoji, net, sectors, analysed,
                      sp_picks, ndx_picks, news, snap):
    today    = datetime.now().strftime("%A, %B %d, %Y")
    time_str = datetime.now().strftime("%I:%M %p ET")

    sent_colors = {
        "Strong Bullish": ("#00d4a1","#003d2e"),
        "Bullish":        ("#3fb950","#0d2b17"),
        "Mildly Bullish": ("#d29922","#2d2208"),
        "Mixed":          ("#f0883e","#2d1600"),
        "Pre-Events":     ("#58a6ff","#0d1e3d"),
        "Bearish":        ("#f85149","#2d0f0e"),
    }
    sc, sbg = sent_colors.get(sentiment, ("#58a6ff","#0d1e3d"))

    # ── Index cards ──
    def idx_card(label, d):
        if not d: return ""
        col   = "#00d4a1" if d["chg"] >= 0 else "#f85149"
        arrow = "▲" if d["chg"] >= 0 else "▼"
        bg    = "rgba(0,212,161,0.06)" if d["chg"] >= 0 else "rgba(248,81,73,0.06)"
        return f"""<div class="idx-card" style="background:{bg};border-color:{col}30">
          <div class="idx-label">{label}</div>
          <div class="idx-price">{d['price']:,.2f}</div>
          <div class="idx-chg" style="color:{col}">{arrow} {d['chg']:+.2f}%</div>
        </div>"""
    idx_cards = "".join(idx_card(l,d) for l,d in snap.items())

    # ── Calendar rows ──
    impact_style = {
        "High":   ("🔴","#da3633","rgba(218,54,51,0.15)"),
        "Medium": ("🟡","#d29922","rgba(210,153,34,0.12)"),
        "Low":    ("🔵","#388bfd","rgba(56,139,253,0.10)"),
    }
    def dir_pill(d):
        pills = {
            "bullish": ("<span class='pill pill-bull'>▲ Bullish</span>"),
            "bearish": ("<span class='pill pill-bear'>▼ Bearish</span>"),
            "pending": ("<span class='pill pill-pend'>⏳ Pending</span>"),
            "neutral": ("<span class='pill pill-neut'>— Neutral</span>"),
        }
        return pills.get(d,"")

    cal_rows = ""
    shown = sorted(analysed, key=lambda x: x["impact"]=="High", reverse=True)[:12]
    for ev in shown:
        dot, ic, ibg = impact_style.get(ev["impact"],("⚪","#555","rgba(0,0,0,0)"))
        a_col = "#00d4a1" if ev["direction"]=="bullish" else ("#f85149" if ev["direction"]=="bearish" else "#8b949e")
        cal_rows += f"""<tr class="cal-row">
          <td class="cal-time">{ev['time']}</td>
          <td class="cal-event">{ev['event']}</td>
          <td><span class="impact-badge" style="color:{ic};background:{ibg};border:1px solid {ic}40">{dot} {ev['impact']}</span></td>
          <td class="cal-actual" style="color:{a_col}">{ev['actual'] if ev['actual'] else '<span class="muted">Pending</span>'}</td>
          <td class="muted">{ev['forecast'] or '—'}</td>
          <td class="muted">{ev['previous'] or '—'}</td>
          <td>{dir_pill(ev['direction'])}</td>
        </tr>"""

    if not cal_rows:
        cal_rows = "<tr><td colspan='7' class='muted' style='text-align:center;padding:20px'>No USD events today</td></tr>"

    # ── Sector tags ──
    sector_tags = "".join(
        f"<span class='sector-tag'>{s}</span>" for s in sectors[:8])

    # ── News list ──
    news_items = "".join(f"<li>{n}</li>" for n in news[:8])

    # ── Stock pick cards ──
    def stock_cards(picks, title, accent):
        if not picks:
            return f"<h2 class='section-title'>{title}</h2><p class='muted'>No strong candidates today.</p>"

        cards = ""
        for i, p in enumerate(picks):
            rank   = i + 1
            top3   = rank <= 3
            pc_col = "#00d4a1" if p["pre_chg"] >= 0 else "#f85149"
            dc_col = "#00d4a1" if p["day_chg"] >= 0 else "#f85149"
            border = accent if top3 else "#1e1e2e"
            glow   = f"box-shadow:0 0 20px {accent}22;" if top3 else ""
            rank_bg = accent if top3 else "#1e1e2e"

            reasons = []
            if abs(p["pre_chg"]) > 0.15: reasons.append(f"Pre-mkt {'▲' if p['pre_chg']>0 else '▼'} {abs(p['pre_chg']):.2f}%")
            if p["vol_ratio"] > 1.4:     reasons.append(f"Vol {p['vol_ratio']:.1f}× avg")
            if p.get("rel_str", 0) > 0.3:reasons.append(f"RS +{p['rel_str']:.1f}% vs index")
            if p["news"]:                reasons.append("News catalyst")
            if not reasons:              reasons.append("Macro tailwind")

            # RSI badge colour
            rsi_val = p.get("rsi", 50)
            rsi_col  = "#00d4a1" if rsi_val < 68 else ("#fbbf24" if rsi_val < 78 else "#f85149")

            # EMA badge
            ema_ok   = p.get("above_ema20", True)
            ema_col  = "#00d4a1" if ema_ok else "#f85149"
            ema_lbl  = "▲ EMA20" if ema_ok else "▼ EMA20"

            news_html = ""
            for n in p["news"][:1]:
                news_html += f'<div class="news-line">▸ {n[:100]}</div>'

            cards += f"""<div class="stock-card" style="border-color:{border};{glow}">
              <div class="card-top">
                <div class="rank-badge" style="background:{rank_bg}">#{rank}</div>
                <div class="card-ticker">{p['ticker']}</div>
                <div class="card-premkt" style="color:{pc_col}">{p['pre_chg']:+.2f}%<span class="card-premkt-label"> pre</span></div>
              </div>
              <div class="card-price">${p['price']:,.2f} &nbsp;·&nbsp;
                <span style="color:{dc_col}">{p['day_chg']:+.2f}% yest</span> &nbsp;·&nbsp;
                <span class="muted">Vol {p['vol_ratio']:.1f}×</span>
              </div>
              <div class="card-signals">
                <span class="sig-badge" style="color:{rsi_col};border-color:{rsi_col}40">RSI {rsi_val:.0f}</span>
                <span class="sig-badge" style="color:{ema_col};border-color:{ema_col}40">{ema_lbl}</span>
              </div>
              <div class="card-reasons">{'  ·  '.join(reasons)}</div>
              {news_html}
              <div class="card-strategy">Wait EMA 9 ✕ EMA 21 on 5m/10m · Trail 1.5%</div>
            </div>"""
        return f"""<div class="picks-section">
          <h2 class="section-title">{title}</h2>
          <div class="picks-grid">{cards}</div>
        </div>"""

    sp_html  = stock_cards(sp_picks,  "S&amp;P 500 — Top 10 Long Candidates", "#58a6ff")
    ndx_html = stock_cards(ndx_picks, "Nasdaq 100 — Top 10 Long Candidates",  "#a371f7")

    # ── Pending banner ──
    pending_html = ""
    if any(e["direction"]=="pending" for e in analysed):
        pending_html = """<div class="pending-banner">
          ⚠️ <strong>Events releasing today</strong> — picks below assume a bullish outcome.
          If data disappoints, wait 5–10 min after release before entering.
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>Market Report — {today}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:       #080b14;
  --surface:  #0d1117;
  --card:     #111827;
  --border:   #1e1e2e;
  --border2:  #2d2d3d;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --bull:     #00d4a1;
  --bear:     #f85149;
  --blue:     #58a6ff;
  --purple:   #a371f7;
  --gold:     #fbbf24;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);
      min-height:100vh;line-height:1.6;font-size:14px}}
/* ── HEADER ── */
.header{{background:linear-gradient(135deg,#0d1117 0%,#111827 50%,#0d1117 100%);
         border-bottom:1px solid var(--border);padding:28px 32px 24px;
         display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px}}
.header-left h1{{font-size:1.5em;font-weight:800;color:#fff;letter-spacing:-0.5px}}
.header-left .subtitle{{color:var(--muted);font-size:0.82em;margin-top:3px}}
.sent-badge{{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;
             border-radius:50px;font-weight:700;font-size:1em;letter-spacing:0.3px;
             background:{sbg};color:{sc};border:1px solid {sc}40}}
/* ── MAIN ── */
.main{{max-width:1400px;margin:0 auto;padding:24px 32px}}
/* ── INDEX BAR ── */
.idx-bar{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px}}
.idx-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;
           padding:16px 20px;transition:transform 0.2s}}
.idx-card:hover{{transform:translateY(-2px)}}
.idx-label{{font-size:0.75em;color:var(--muted);font-weight:600;text-transform:uppercase;
            letter-spacing:0.8px;margin-bottom:6px}}
.idx-price{{font-family:'JetBrains Mono',monospace;font-size:1.25em;font-weight:600;
            color:#fff;margin-bottom:3px}}
.idx-chg{{font-family:'JetBrains Mono',monospace;font-size:0.88em;font-weight:600}}
/* ── SECTIONS ── */
.section{{background:var(--card);border:1px solid var(--border);border-radius:14px;
          padding:24px;margin-bottom:20px}}
.section-title{{font-size:1.05em;font-weight:700;color:#fff;margin-bottom:16px;
                display:flex;align-items:center;gap:8px}}
.section-title::before{{content:'';display:inline-block;width:3px;height:18px;
                         border-radius:2px;background:var(--blue)}}
/* ── CALENDAR ── */
table{{width:100%;border-collapse:collapse}}
.cal-row{{border-bottom:1px solid var(--border)}}
.cal-row:last-child{{border-bottom:none}}
.cal-row:hover{{background:rgba(255,255,255,0.02)}}
th{{font-size:0.72em;font-weight:600;color:var(--muted);text-transform:uppercase;
    letter-spacing:0.7px;padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}}
td{{padding:11px 10px;vertical-align:middle}}
.cal-time{{font-family:'JetBrains Mono',monospace;font-size:0.8em;color:var(--muted);white-space:nowrap}}
.cal-event{{font-weight:500;color:var(--text)}}
.cal-actual{{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:0.95em}}
.impact-badge{{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;
               border-radius:20px;font-size:0.72em;font-weight:600;white-space:nowrap}}
.pill{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.75em;font-weight:600}}
.pill-bull{{background:rgba(0,212,161,0.15);color:var(--bull)}}
.pill-bear{{background:rgba(248,81,73,0.15);color:var(--bear)}}
.pill-pend{{background:rgba(88,166,255,0.12);color:var(--blue)}}
.pill-neut{{background:rgba(100,116,139,0.15);color:var(--muted)}}
/* ── SENTIMENT ── */
.sentiment-row{{display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
.sent-score{{font-family:'JetBrains Mono',monospace;font-size:0.85em;color:var(--muted);margin-top:8px}}
.sector-tag{{display:inline-block;background:rgba(88,166,255,0.1);color:var(--blue);
             border:1px solid rgba(88,166,255,0.25);padding:3px 11px;border-radius:20px;
             font-size:0.78em;font-weight:500;margin:3px}}
.sectors-wrap{{margin-top:10px}}
/* ── NEWS ── */
.news-list{{list-style:none;padding:0}}
.news-list li{{padding:8px 0;border-bottom:1px solid var(--border);font-size:0.88em;
               color:#94a3b8;display:flex;gap:10px}}
.news-list li::before{{content:'▸';color:var(--blue);flex-shrink:0}}
.news-list li:last-child{{border-bottom:none}}
/* ── PICKS ── */
.picks-section{{margin-bottom:20px}}
.picks-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.stock-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;
             padding:16px;transition:transform 0.15s,box-shadow 0.15s}}
.stock-card:hover{{transform:translateY(-2px)}}
.card-top{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.rank-badge{{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;
             justify-content:center;font-size:0.75em;font-weight:800;color:#fff;flex-shrink:0}}
.card-ticker{{font-size:1.3em;font-weight:800;color:#fff;flex:1}}
.card-premkt{{font-family:'JetBrains Mono',monospace;font-size:1.05em;font-weight:700;text-align:right}}
.card-premkt-label{{font-size:0.65em;color:var(--muted);font-weight:400}}
.card-price{{font-family:'JetBrains Mono',monospace;font-size:0.8em;color:var(--muted);margin-bottom:7px}}
.card-signals{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:7px}}
.sig-badge{{font-family:'JetBrains Mono',monospace;font-size:0.72em;font-weight:600;
             padding:2px 8px;border-radius:6px;border:1px solid;letter-spacing:0.3px}}
.card-reasons{{font-size:0.8em;color:#94a3b8;margin-bottom:5px}}
.news-line{{font-size:0.76em;color:var(--muted);border-left:2px solid var(--border2);
            padding-left:8px;margin:4px 0;line-height:1.4}}
.card-strategy{{font-size:0.73em;color:rgba(0,212,161,0.7);margin-top:8px;
                padding-top:7px;border-top:1px solid var(--border)}}
/* ── PENDING BANNER ── */
.pending-banner{{background:rgba(88,166,255,0.08);border:1px solid rgba(88,166,255,0.25);
                 border-radius:10px;padding:12px 16px;font-size:0.87em;color:#93c5fd;margin-bottom:16px}}
/* ── TRADING RULES ── */
.rules-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}}
.rule-item{{background:var(--surface);border:1px solid var(--border);border-radius:10px;
            padding:12px 14px;font-size:0.83em;color:#94a3b8;display:flex;gap:10px}}
.rule-icon{{font-size:1.1em;flex-shrink:0}}
/* ── FOOTER ── */
.footer{{text-align:center;color:var(--muted);font-size:0.75em;padding:24px 32px;
         border-top:1px solid var(--border);margin-top:8px}}
.muted{{color:var(--muted)}}
/* ── RESPONSIVE ── */
@media(max-width:768px){{
  .header{{padding:20px 16px}}
  .main{{padding:16px}}
  .idx-bar{{grid-template-columns:repeat(2,1fr)}}
  .picks-grid{{grid-template-columns:1fr}}
  .rules-grid{{grid-template-columns:1fr}}
  th,td{{padding:8px 6px}}
}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <h1>🌅 Morning Market Report</h1>
    <div class="subtitle">{today} &nbsp;·&nbsp; {time_str} &nbsp;·&nbsp; EMA 9/21 &nbsp;·&nbsp; Long Only &nbsp;·&nbsp; 1.5% Trail</div>
  </div>
  <div class="sent-badge">{emoji} {sentiment}</div>
</div>

<div class="main">

  <!-- INDEX SNAPSHOT -->
  <div class="idx-bar">{idx_cards}</div>

  <!-- ECONOMIC CALENDAR -->
  <div class="section">
    <div class="section-title">US Economic Calendar</div>
    {pending_html}
    <table>
      <tr>
        <th>Time</th><th>Event</th><th>Impact</th>
        <th>Actual</th><th>Forecast</th><th>Previous</th><th>Signal</th>
      </tr>
      {cal_rows}
    </table>
  </div>

  <!-- SENTIMENT -->
  <div class="section">
    <div class="section-title">Market Sentiment</div>
    <div class="sentiment-row">
      <div class="sent-badge">{emoji} {sentiment}</div>
    </div>
    <div class="sent-score">Macro score: {net:+d} &nbsp;·&nbsp; Favoured sectors:</div>
    <div class="sectors-wrap">{sector_tags}</div>
  </div>

  <!-- HEADLINES -->
  <div class="section">
    <div class="section-title">Market Headlines</div>
    <ul class="news-list">{"".join(f"<li>{n}</li>" for n in news[:8])}</ul>
  </div>

  <!-- S&P 500 PICKS -->
  <div class="section">
    {sp_html}
  </div>

  <!-- NDX PICKS -->
  <div class="section">
    {ndx_html}
  </div>

  <!-- TRADING RULES -->
  <div class="section">
    <div class="section-title">Trading Rules</div>
    <div class="rules-grid">
      <div class="rule-item"><span class="rule-icon">📍</span><span><strong>Long only</strong> — only enter when EMA 9 crosses above EMA 21</span></div>
      <div class="rule-item"><span class="rule-icon">⏱</span><span>Wait for the <strong>first 5-min bar to close</strong> before entering any trade</span></div>
      <div class="rule-item"><span class="rule-icon">🛡</span><span><strong>1.5% trailing stop</strong> on every position — no exceptions</span></div>
      <div class="rule-item"><span class="rule-icon">📰</span><span>High-impact events during session → <strong>wait 5–10 min</strong> after the release</span></div>
      <div class="rule-item"><span class="rule-icon">📊</span><span>Pre-market movers with <strong>vol 1.5×+</strong> tend to continue first 30–60 min</span></div>
      <div class="rule-item"><span class="rule-icon">⚡</span><span>VIX &gt; 20 = reduce size. VIX &lt; 15 = normal size, cleaner trends</span></div>
    </div>
  </div>

</div>

<div class="footer">
  Auto-generated by Claude Code &nbsp;·&nbsp; Data: Trading Economics + Yahoo Finance
  &nbsp;·&nbsp; Refreshes every 5 min &nbsp;·&nbsp; Not financial advice
</div>

</body>
</html>"""


# ─── SUMMARY EMAIL ────────────────────────────────────────────────────────────
def build_email(sentiment, emoji, sp_picks, ndx_picks, snap, net, report_url):
    today    = datetime.now().strftime("%A, %b %d")
    sent_col = {"Strong Bullish":"#00d4a1","Bullish":"#3fb950","Mildly Bullish":"#d29922",
                "Mixed":"#f0883e","Pre-Events":"#58a6ff","Bearish":"#f85149"}.get(sentiment,"#58a6ff")

    def snap_row(label, d):
        if not d: return ""
        col   = "#00d4a1" if d["chg"]>=0 else "#f85149"
        arrow = "▲" if d["chg"]>=0 else "▼"
        return (f"<td style='padding:8px 16px 8px 0;font-family:monospace'>"
                f"<span style='color:#64748b;font-size:11px'>{label}</span><br>"
                f"<strong>{d['price']:,.0f}</strong> "
                f"<span style='color:{col}'>{arrow}{d['chg']:+.2f}%</span></td>")

    def pick_rows(picks):
        rows = ""
        for i, p in enumerate(picks[:5]):
            col = "#00d4a1" if p["pre_chg"]>=0 else "#f85149"
            rows += (f"<tr><td style='padding:5px 0;color:#94a3b8;font-size:12px'>#{i+1}</td>"
                     f"<td style='padding:5px 8px;font-weight:700;color:#fff;font-size:14px'>{p['ticker']}</td>"
                     f"<td style='padding:5px 0;font-family:monospace;color:{col};font-size:13px'>{p['pre_chg']:+.2f}%</td>"
                     f"<td style='padding:5px 0 5px 12px;color:#64748b;font-size:12px'>${p['price']:,.2f}</td></tr>")
        return rows

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#080b14;font-family:Inter,-apple-system,sans-serif">
<div style="max-width:600px;margin:0 auto;padding:20px">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#0d1117,#111827);border:1px solid #1e1e2e;
              border-radius:16px;padding:28px;margin-bottom:16px;text-align:center">
    <div style="font-size:13px;color:#64748b;margin-bottom:6px">🌅 MORNING MARKET REPORT</div>
    <div style="font-size:22px;font-weight:800;color:#fff;margin-bottom:12px">{today}</div>
    <div style="display:inline-block;background:rgba(0,0,0,0.3);border:1px solid {sent_col}40;
                border-radius:50px;padding:8px 20px;font-weight:700;color:{sent_col};font-size:15px">
      {emoji} {sentiment}
    </div>
  </div>

  <!-- Index Snapshot -->
  <div style="background:#0d1117;border:1px solid #1e1e2e;border-radius:12px;
              padding:16px 20px;margin-bottom:16px">
    <div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:10px">Index Snapshot</div>
    <table style="width:100%;border-collapse:collapse">
      <tr>{"".join(snap_row(l,d) for l,d in snap.items())}</tr>
    </table>
  </div>

  <!-- S&P 500 Picks -->
  <div style="background:#0d1117;border:1px solid #1e1e2e;border-radius:12px;
              padding:16px 20px;margin-bottom:12px">
    <div style="font-size:11px;font-weight:600;color:#58a6ff;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:10px">S&amp;P 500 Top 5</div>
    <table style="width:100%;border-collapse:collapse">{pick_rows(sp_picks)}</table>
  </div>

  <!-- NDX Picks -->
  <div style="background:#0d1117;border:1px solid #1e1e2e;border-radius:12px;
              padding:16px 20px;margin-bottom:20px">
    <div style="font-size:11px;font-weight:600;color:#a371f7;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:10px">Nasdaq 100 Top 5</div>
    <table style="width:100%;border-collapse:collapse">{pick_rows(ndx_picks)}</table>
  </div>

  <!-- CTA Button -->
  <div style="text-align:center;margin-bottom:20px">
    <a href="{report_url}" target="_blank" style="display:inline-block;background:#1d4ed8;color:#ffffff;text-decoration:none;font-weight:700;font-size:16px;padding:16px 40px;border-radius:50px;letter-spacing:0.3px;mso-padding-alt:0">📊 Open Full Report →</a>
    <div style="margin-top:10px;font-size:11px;color:#475569">Full economic calendar · All 10 picks · Trading rules</div>
    <div style="margin-top:6px;font-size:11px;color:#475569">Or copy: <a href="{report_url}" style="color:#58a6ff">{report_url}</a></div>
  </div>

  <!-- Footer -->
  <div style="text-align:center;color:#334155;font-size:11px;padding-top:12px;
              border-top:1px solid #1e1e2e">
    Auto-generated by Claude Code &nbsp;·&nbsp; Not financial advice
  </div>

</div></body></html>"""


# ─── EMAIL SENDER ─────────────────────────────────────────────────────────────
def send_email(html, subject):
    s = os.environ.get("GMAIL_USER","")
    p = os.environ.get("GMAIL_APP_PASSWORD","")
    r = os.environ.get("REPORT_EMAIL","")
    if not all([s,p,r]): return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Market Report <{s}>"
    msg["To"]      = r
    msg.attach(MIMEText(html,"html","utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com",465) as srv:
        srv.login(s,p)
        srv.sendmail(s,r,msg.as_string())
    return True


# ─── DST CHECK ────────────────────────────────────────────────────────────────
def check_et_time():
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.hour != 8 or now.minute < 28:
            print(f"Not 8:30 AM ET ({now.strftime('%I:%M %p ET')}), skipping.")
            sys.exit(0)
    except: pass


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if CLOUD_MODE:
        check_et_time()

    print(f"\n{'='*62}")
    print(f"  MACRO REPORT  {datetime.now().strftime('%A %b %d, %Y  %I:%M %p')}")
    print(f"  Mode: {'CLOUD' if CLOUD_MODE else 'LOCAL'}")
    print(f"{'='*62}")

    print("\n  [1/6] Index snapshot...")
    snap = index_snapshot()
    for l, d in snap.items():
        if d: print(f"    {l}: {d['price']:>10,.2f}  ({d['chg']:+.2f}%)")

    print("\n  [2/6] Economic calendar...")
    events = scrape_calendar()
    print(f"    {len(events)} events found")

    print("\n  [3/6] Sentiment analysis...")
    sentiment, emoji, net, sectors, analysed = get_sentiment(events)
    print(f"    {emoji} {sentiment}  (score {net:+d})")

    print("\n  [4/6] Market news...")
    news = get_market_news()
    print(f"    {len(news)} headlines")

    print("\n  [5/6] Stock selection...")
    sp_picks, ndx_picks = pick_stocks(sectors)
    print(f"    S&P:  {', '.join(p['ticker'] for p in sp_picks)}")
    print(f"    NDX:  {', '.join(p['ticker'] for p in ndx_picks)}")

    print("\n  [6/6] Building & delivering report...")
    full_html  = build_full_report(events, sentiment, emoji, net, sectors,
                                   analysed, sp_picks, ndx_picks, news, snap)
    email_html = build_email(sentiment, emoji, sp_picks, ndx_picks, snap, net, REPORT_URL)
    subject    = f"📊 {sentiment} | {', '.join(p['ticker'] for p in sp_picks[:3])} | {datetime.now().strftime('%a %b %d')}"

    # Always save the full report to docs/index.html
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR,"index.html"),"w",encoding="utf-8") as f:
        f.write(full_html)
    print(f"    Full report → {OUTPUT_DIR}/index.html")

    if CLOUD_MODE:
        send_email(email_html, subject)
        print(f"    Email sent → {os.environ.get('REPORT_EMAIL','')}")
    else:
        import webbrowser
        webbrowser.open(f"file:///{os.path.abspath(os.path.join(OUTPUT_DIR,'index.html'))}")
        print("    Opened in browser.")

    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    main()
