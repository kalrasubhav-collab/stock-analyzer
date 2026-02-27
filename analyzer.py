"""
NSE/BSE Stock Analyzer — Phase 1 CLI Tool
AI-powered stock analysis for Indian markets using yfinance + Claude API.
For educational purposes only — not financial advice.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
import anthropic


# ──────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────

def fetch_stock_data(symbol: str) -> tuple[pd.DataFrame, dict]:
    """Fetch historical OHLCV data and fundamental info from yfinance."""
    print(f"\n⏳ Fetching data for {symbol}...")

    ticker = yf.Ticker(symbol)

    # 1 year of daily history — enough for 200-day MA
    hist = ticker.history(period="1y")
    if hist.empty:
        print(f"❌ No data found for '{symbol}'. Check the symbol and try again.")
        print("   Examples: RELIANCE.NS  TCS.NS  INFY.NS  HDFCBANK.NS")
        sys.exit(1)

    info = ticker.info
    print(f"✅ Got {len(hist)} trading days of data.")
    return hist, info


# ──────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────

def calculate_moving_averages(df: pd.DataFrame) -> dict:
    """50-day and 200-day Simple Moving Averages."""
    close = df["Close"]
    ma50  = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    current_price = close.iloc[-1]

    return {
        "current_price": round(float(current_price), 2),
        "ma50":          round(float(ma50), 2)  if not pd.isna(ma50)  else None,
        "ma200":         round(float(ma200), 2) if not pd.isna(ma200) else None,
        "price_vs_ma50":  round(float((current_price - ma50)  / ma50  * 100), 2) if not pd.isna(ma50)  else None,
        "price_vs_ma200": round(float((current_price - ma200) / ma200 * 100), 2) if not pd.isna(ma200) else None,
        "golden_cross": (not pd.isna(ma50) and not pd.isna(ma200) and ma50 > ma200),
    }


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    """14-day RSI."""
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = round(float(rsi.iloc[-1]), 2)

    if current_rsi >= 70:
        signal = "OVERBOUGHT"
    elif current_rsi <= 30:
        signal = "OVERSOLD"
    else:
        signal = "NEUTRAL"

    return {"rsi": current_rsi, "signal": signal}


def calculate_macd(df: pd.DataFrame) -> dict:
    """MACD (12, 26, 9)."""
    close  = df["Close"]
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal

    return {
        "macd":      round(float(macd.iloc[-1]), 4),
        "signal":    round(float(signal.iloc[-1]), 4),
        "histogram": round(float(hist.iloc[-1]), 4),
        "bullish":   bool(macd.iloc[-1] > signal.iloc[-1]),
    }


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20) -> dict:
    """20-day Bollinger Bands (±2 std dev)."""
    close = df["Close"]
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    current = close.iloc[-1]
    u = upper.iloc[-1]
    l = lower.iloc[-1]
    m = sma.iloc[-1]
    bandwidth = round(float((u - l) / m * 100), 2) if not pd.isna(m) else None

    if current > u:
        position = "ABOVE_UPPER (overbought)"
    elif current < l:
        position = "BELOW_LOWER (oversold)"
    else:
        pct = round(float((current - l) / (u - l) * 100), 2) if (u - l) != 0 else 50
        position = f"WITHIN_BANDS ({pct}% from lower)"

    return {
        "upper":     round(float(u), 2) if not pd.isna(u) else None,
        "middle":    round(float(m), 2) if not pd.isna(m) else None,
        "lower":     round(float(l), 2) if not pd.isna(l) else None,
        "bandwidth": bandwidth,
        "position":  position,
    }


def calculate_volume_analysis(df: pd.DataFrame) -> dict:
    """Compare latest volume to 20-day average."""
    volume     = df["Volume"]
    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    latest_vol = volume.iloc[-1]
    ratio      = round(float(latest_vol / avg_vol_20), 2) if not pd.isna(avg_vol_20) else None

    return {
        "latest_volume":     int(latest_vol),
        "avg_volume_20d":    int(avg_vol_20) if not pd.isna(avg_vol_20) else None,
        "volume_ratio":      ratio,
        "above_average":     ratio > 1.0 if ratio else None,
    }


def get_all_technicals(df: pd.DataFrame) -> dict:
    """Run all technical indicator calculations."""
    print("📐 Calculating technical indicators...")
    return {
        "moving_averages":   calculate_moving_averages(df),
        "rsi":               calculate_rsi(df),
        "macd":              calculate_macd(df),
        "bollinger_bands":   calculate_bollinger_bands(df),
        "volume_analysis":   calculate_volume_analysis(df),
    }


# ──────────────────────────────────────────────
# FUNDAMENTAL DATA
# ──────────────────────────────────────────────

def _safe(info: dict, *keys, default=None, fmt=None):
    """Try multiple keys, return first found value (optionally formatted)."""
    for key in keys:
        val = info.get(key)
        if val is not None and val != "N/A" and val != 0:
            if fmt == "crore" and isinstance(val, (int, float)):
                return round(val / 1e7, 2)   # convert to INR crore
            if fmt == "pct" and isinstance(val, (int, float)):
                return round(val * 100, 2)
            if isinstance(val, float):
                return round(val, 2)
            return val
    return default


def extract_fundamentals(info: dict) -> dict:
    """Pull key fundamental metrics from yfinance info dict."""
    print("📊 Extracting fundamental data...")

    market_cap_raw = info.get("marketCap")
    market_cap_cr  = round(market_cap_raw / 1e7, 2) if market_cap_raw else None

    return {
        "company_name":    info.get("longName") or info.get("shortName", "N/A"),
        "sector":          info.get("sector", "N/A"),
        "industry":        info.get("industry", "N/A"),
        "currency":        info.get("currency", "INR"),
        "pe_ratio":        _safe(info, "trailingPE", "forwardPE"),
        "pb_ratio":        _safe(info, "priceToBook"),
        "market_cap_cr":   market_cap_cr,
        "fifty_two_week_high": _safe(info, "fiftyTwoWeekHigh"),
        "fifty_two_week_low":  _safe(info, "fiftyTwoWeekLow"),
        "revenue_growth":  _safe(info, "revenueGrowth", fmt="pct"),     # % YoY
        "earnings_growth": _safe(info, "earningsGrowth", fmt="pct"),    # % YoY
        "debt_to_equity":  _safe(info, "debtToEquity"),
        "roe":             _safe(info, "returnOnEquity", fmt="pct"),     # %
        "roa":             _safe(info, "returnOnAssets", fmt="pct"),     # %
        "current_ratio":   _safe(info, "currentRatio"),
        "dividend_yield":  _safe(info, "dividendYield", fmt="pct"),     # %
        "beta":            _safe(info, "beta"),
        "analyst_target":  _safe(info, "targetMeanPrice"),
        "analyst_rating":  info.get("recommendationKey", "N/A"),
    }


# ──────────────────────────────────────────────
# CLAUDE AI ANALYSIS
# ──────────────────────────────────────────────

def build_analysis_prompt(symbol: str, technicals: dict, fundamentals: dict) -> str:
    """Construct the detailed prompt for Claude."""
    ma   = technicals["moving_averages"]
    rsi  = technicals["rsi"]
    macd = technicals["macd"]
    bb   = technicals["bollinger_bands"]
    vol  = technicals["volume_analysis"]
    f    = fundamentals

    prompt = f"""You are an expert Indian stock market analyst specialising in NSE/BSE equities.
Analyse the following data for {symbol} ({f['company_name']}) and provide a structured recommendation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Price      : {ma['current_price']} {f['currency']}
50-Day MA          : {ma['ma50']}   (Price is {ma['price_vs_ma50']}% above/below)
200-Day MA         : {ma['ma200']}  (Price is {ma['price_vs_ma200']}% above/below)
Golden Cross       : {'YES — bullish' if ma['golden_cross'] else 'NO — bearish'}

RSI (14-day)       : {rsi['rsi']} → {rsi['signal']}

MACD               : {macd['macd']}
MACD Signal Line   : {macd['signal']}
MACD Histogram     : {macd['histogram']}
MACD Trend         : {'Bullish (MACD above signal)' if macd['bullish'] else 'Bearish (MACD below signal)'}

Bollinger Bands (20-day):
  Upper Band       : {bb['upper']}
  Middle Band      : {bb['middle']}
  Lower Band       : {bb['lower']}
  Bandwidth        : {bb['bandwidth']}%
  Price Position   : {bb['position']}

Volume Analysis:
  Latest Volume    : {vol['latest_volume']:,}
  20-Day Avg Vol   : {vol['avg_volume_20d']:,}
  Volume Ratio     : {vol['volume_ratio']}x  ({'above' if vol['above_average'] else 'below'} average)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTAL DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sector / Industry  : {f['sector']} / {f['industry']}
Market Cap         : ₹{f['market_cap_cr']:,} Cr  (approx)
52-Week High       : {f['fifty_two_week_high']}
52-Week Low        : {f['fifty_two_week_low']}

P/E Ratio          : {f['pe_ratio']}
P/B Ratio          : {f['pb_ratio']}
Revenue Growth YoY : {f['revenue_growth']}%
Earnings Growth YoY: {f['earnings_growth']}%
Debt to Equity     : {f['debt_to_equity']}
Return on Equity   : {f['roe']}%
Return on Assets   : {f['roa']}%
Current Ratio      : {f['current_ratio']}
Dividend Yield     : {f['dividend_yield']}%
Beta               : {f['beta']}
Analyst Target     : {f['analyst_target']} {f['currency']}
Analyst Rating     : {f['analyst_rating']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Please provide a comprehensive analysis structured EXACTLY as follows:

## SHORT-TERM VERDICT (1–4 Weeks)
**Rating:** BUY / HOLD / SELL
**Reasoning:** (3–5 bullet points based on technical indicators)
**Entry Zone:** (price range or "wait for dip to X")
**Stop Loss:** (price level)
**Target:** (price target within 1–4 weeks)

## LONG-TERM VERDICT (1–3 Years)
**Rating:** BUY / HOLD / SELL
**Reasoning:** (3–5 bullet points based on fundamentals + technicals)
**Entry Strategy:** (lump sum / SIP / wait for correction)
**Price Target (1 Year):** (if determinable)
**Price Target (3 Years):** (if determinable)

## KEY STRENGTHS
(3–5 bullet points highlighting the stock's positives)

## KEY RISKS
(3–5 bullet points with specific risks to watch)

## SUMMARY
One paragraph (4–5 sentences) summarising the overall investment case.

Important: Be specific, data-driven, and mention actual numbers from the data above.
Disclaimer: This is for educational purposes only and not financial advice."""

    return prompt


def get_claude_analysis(symbol: str, technicals: dict, fundamentals: dict) -> str:
    """Send data to Claude API and return the analysis text."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    print("🤖 Sending data to Claude AI for analysis...")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_analysis_prompt(symbol, technicals, fundamentals)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


# ──────────────────────────────────────────────
# PRETTY PRINTING
# ──────────────────────────────────────────────

DIVIDER = "═" * 60

def print_header(symbol: str, fundamentals: dict):
    print(f"\n{DIVIDER}")
    print(f"  📈  {fundamentals['company_name']}  ({symbol})")
    print(f"  🏭  {fundamentals['sector']} › {fundamentals['industry']}")
    print(DIVIDER)


def print_technicals(technicals: dict, fundamentals: dict):
    ma   = technicals["moving_averages"]
    rsi  = technicals["rsi"]
    macd = technicals["macd"]
    bb   = technicals["bollinger_bands"]
    vol  = technicals["volume_analysis"]
    cur  = fundamentals["currency"]

    print("\n📐  TECHNICAL INDICATORS")
    print("─" * 60)
    print(f"  💰 Current Price   : {ma['current_price']} {cur}")
    print(f"  📉 50-Day MA       : {ma['ma50']}  ({'+' if (ma['price_vs_ma50'] or 0) > 0 else ''}{ma['price_vs_ma50']}% vs price)")
    print(f"  📉 200-Day MA      : {ma['ma200']} ({'+' if (ma['price_vs_ma200'] or 0) > 0 else ''}{ma['price_vs_ma200']}% vs price)")
    cross = "🌟 Golden Cross (BULLISH)" if ma["golden_cross"] else "💀 Death Cross (BEARISH)"
    print(f"  ⚡ MA Cross Signal : {cross}")

    rsi_emoji = "🔴" if rsi["rsi"] >= 70 else ("🟢" if rsi["rsi"] <= 30 else "🟡")
    print(f"\n  {rsi_emoji} RSI (14-day)     : {rsi['rsi']} → {rsi['signal']}")

    macd_emoji = "🟢" if macd["bullish"] else "🔴"
    print(f"\n  {macd_emoji} MACD            : {macd['macd']}")
    print(f"     Signal Line    : {macd['signal']}")
    print(f"     Histogram      : {macd['histogram']}  ({'Bullish momentum' if macd['bullish'] else 'Bearish momentum'})")

    print(f"\n  📊 Bollinger Bands (20-day)")
    print(f"     Upper          : {bb['upper']}")
    print(f"     Middle         : {bb['middle']}")
    print(f"     Lower          : {bb['lower']}")
    print(f"     Bandwidth      : {bb['bandwidth']}%")
    print(f"     Position       : {bb['position']}")

    vol_emoji = "📈" if vol["above_average"] else "📉"
    print(f"\n  {vol_emoji} Volume Analysis")
    print(f"     Latest         : {vol['latest_volume']:,}")
    print(f"     20-Day Avg     : {vol['avg_volume_20d']:,}")
    print(f"     Ratio          : {vol['volume_ratio']}x average")


def print_fundamentals(fundamentals: dict):
    f = fundamentals

    def _fmt(val, suffix="", prefix=""):
        return f"{prefix}{val}{suffix}" if val is not None else "N/A"

    print(f"\n\n📊  FUNDAMENTAL DATA")
    print("─" * 60)
    print(f"  🏦 Market Cap      : {_fmt(f['market_cap_cr'], ' Cr', '₹')}")
    print(f"  📅 52-Week High    : {_fmt(f['fifty_two_week_high'])}")
    print(f"  📅 52-Week Low     : {_fmt(f['fifty_two_week_low'])}")
    print(f"  💹 P/E Ratio       : {_fmt(f['pe_ratio'])}")
    print(f"  💹 P/B Ratio       : {_fmt(f['pb_ratio'])}")
    print(f"  📈 Revenue Growth  : {_fmt(f['revenue_growth'], '%')}")
    print(f"  📈 Earnings Growth : {_fmt(f['earnings_growth'], '%')}")
    print(f"  🏋️  Debt / Equity  : {_fmt(f['debt_to_equity'])}")
    print(f"  💰 ROE             : {_fmt(f['roe'], '%')}")
    print(f"  💰 ROA             : {_fmt(f['roa'], '%')}")
    print(f"  🔄 Current Ratio   : {_fmt(f['current_ratio'])}")
    print(f"  💵 Dividend Yield  : {_fmt(f['dividend_yield'], '%')}")
    print(f"  📡 Beta            : {_fmt(f['beta'])}")
    print(f"  🎯 Analyst Target  : {_fmt(f['analyst_target'])}")
    print(f"  ⭐ Analyst Rating  : {str(f['analyst_rating']).upper()}")


def print_ai_analysis(analysis: str):
    print(f"\n\n🤖  AI ANALYSIS  (powered by Claude)")
    print("─" * 60)
    print(analysis)


def print_disclaimer():
    print(f"\n{DIVIDER}")
    print("  ⚠️   DISCLAIMER")
    print("  This analysis is for educational purposes only.")
    print("  It is NOT financial advice. Always do your own")
    print("  research before making any investment decisions.")
    print(DIVIDER)


# ──────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🇮🇳  NSE/BSE AI Stock Analyzer  🇮🇳")
    print("=" * 60)
    print("  Powered by yfinance + Claude AI")
    print("  For educational purposes only.\n")

    # Get symbol from arg or interactive input
    if len(sys.argv) > 1:
        symbol = sys.argv[1].strip().upper()
    else:
        symbol = input("  Enter stock symbol (e.g. RELIANCE.NS, TCS.NS): ").strip().upper()

    if not symbol:
        print("❌ No symbol entered. Exiting.")
        sys.exit(1)

    # Normalise: if user typed just "RELIANCE" suggest adding .NS
    if "." not in symbol:
        print(f"  ℹ️  Tip: For NSE use {symbol}.NS, for BSE use {symbol}.BO")
        print(f"  Using {symbol}.NS by default.")
        symbol = symbol + ".NS"

    # Run pipeline
    hist, info = fetch_stock_data(symbol)
    technicals  = get_all_technicals(hist)
    fundamentals = extract_fundamentals(info)
    analysis    = get_claude_analysis(symbol, technicals, fundamentals)

    # Print results
    print_header(symbol, fundamentals)
    print_technicals(technicals, fundamentals)
    print_fundamentals(fundamentals)
    print_ai_analysis(analysis)
    print_disclaimer()


if __name__ == "__main__":
    main()
