"""
NSE/BSE Stock Analyzer — Phase 2 Web Interface
Flask app powered by analyzer.py + Claude AI.
For educational purposes only — not financial advice.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
from flask import Flask, render_template, request

from analyzer import (
    get_all_technicals,
    extract_fundamentals,
    get_claude_analysis,
)

app = Flask(__name__)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def fetch_data_safe(symbol: str):
    """Web-safe fetch: raises ValueError instead of sys.exit on bad symbol."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y")
    if hist.empty:
        raise ValueError(
            f"No data found for '{symbol}'. "
            "Please check the symbol — e.g. RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS"
        )
    return hist, ticker.info


def parse_verdicts(text: str) -> dict:
    """Extract BUY / HOLD / SELL verdicts from Claude's structured response."""
    verdicts = {"short_term": "N/A", "long_term": "N/A"}
    current_section = None
    for line in text.split("\n"):
        u = line.upper()
        if "SHORT-TERM VERDICT" in u:
            current_section = "short"
        elif "LONG-TERM VERDICT" in u:
            current_section = "long"
        if current_section and "**RATING:**" in u:
            for v in ["BUY", "HOLD", "SELL"]:
                if v in u:
                    key = "short_term" if current_section == "short" else "long_term"
                    if verdicts[key] == "N/A":
                        verdicts[key] = v
                    break
    return verdicts


# ──────────────────────────────────────────────
# TEMPLATE FILTERS
# ──────────────────────────────────────────────

@app.template_filter("fmt_num")
def fmt_num(val, decimals=2):
    if val is None:
        return "N/A"
    try:
        return f"{float(val):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


@app.template_filter("fmt_crore")
def fmt_crore(val):
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if v >= 100_000:
            return f"₹{v / 100_000:.2f}L Cr"
        if v >= 1_000:
            return f"₹{v / 1_000:.1f}K Cr"
        return f"₹{v:,.0f} Cr"
    except (TypeError, ValueError):
        return "N/A"


@app.template_filter("fmt_pct")
def fmt_pct(val):
    if val is None:
        return "N/A"
    try:
        v = float(val)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


@app.template_filter("clamp")
def clamp_filter(val, lo=0, hi=100):
    if val is None:
        return lo
    return max(lo, min(hi, float(val)))


@app.template_filter("fmt_vol")
def fmt_vol(val):
    if val is None:
        return "N/A"
    try:
        v = int(val)
        if v >= 10_000_000:
            return f"{v / 10_000_000:.2f} Cr"
        if v >= 100_000:
            return f"{v / 100_000:.2f} L"
        return f"{v:,}"
    except (TypeError, ValueError):
        return "N/A"


@app.template_filter("verdict_class")
def verdict_class(verdict: str) -> str:
    return {
        "BUY":  "verdict-buy",
        "HOLD": "verdict-hold",
        "SELL": "verdict-sell",
    }.get(str(verdict).upper(), "verdict-na")


@app.template_filter("pct_class")
def pct_class(val):
    try:
        return "positive" if float(val) > 0 else ("negative" if float(val) < 0 else "neutral")
    except (TypeError, ValueError):
        return "neutral"


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    symbol = request.form.get("symbol", "").strip().upper()

    if not symbol:
        return render_template("index.html", error="Please enter a stock symbol.")

    if "." not in symbol:
        symbol = symbol + ".NS"

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return render_template(
            "index.html",
            error="ANTHROPIC_API_KEY environment variable is not configured on the server.",
            symbol=symbol,
        )

    try:
        hist, info = fetch_data_safe(symbol)
        technicals   = get_all_technicals(hist)
        fundamentals = extract_fundamentals(info)
        analysis     = get_claude_analysis(symbol, technicals, fundamentals)
        verdicts     = parse_verdicts(analysis)

        # 52-week range position (0–100 %) for the visual range slider
        current = technicals["moving_averages"]["current_price"]
        high52  = fundamentals.get("fifty_two_week_high")
        low52   = fundamentals.get("fifty_two_week_low")
        if high52 and low52 and (high52 - low52) > 0:
            position_52w = round((current - low52) / (high52 - low52) * 100, 1)
        else:
            position_52w = None

        return render_template(
            "index.html",
            symbol=symbol,
            technicals=technicals,
            fundamentals=fundamentals,
            analysis=analysis,
            verdicts=verdicts,
            position_52w=position_52w,
        )

    except SystemExit:
        return render_template(
            "index.html",
            error="A system error occurred. Please verify your API key and try again.",
            symbol=symbol,
        )
    except ValueError as e:
        return render_template("index.html", error=str(e), symbol=symbol)
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Analysis failed: {e}",
            symbol=symbol,
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
