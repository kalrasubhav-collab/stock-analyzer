# NSE/BSE Stock Analyzer

## Project Goal
An AI-powered stock analyzer for Indian markets (NSE/BSE) that helps
investors pick stocks for short-term and long-term investment using
technical and fundamental analysis.

## Tech Stack
- Python 3.13
- yfinance (fetch real stock data from NSE/BSE)
- pandas (data analysis)
- Anthropic Claude API (AI-powered recommendations)
- Flask (web interface - Phase 2)

## Stock Symbol Format
- NSE stocks: add .NS suffix (e.g., RELIANCE.NS, TCS.NS, INFY.NS)
- BSE stocks: add .BO suffix (e.g., RELIANCE.BO, TCS.BO)

## Phase 1 — CLI Tool (analyzer.py)
Build a command line tool that:
1. Takes a stock symbol as input (e.g., RELIANCE.NS)
2. Fetches real data using yfinance
3. Calculates technical indicators:
   - 50-day and 200-day Moving Averages
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Volume analysis
4. Extracts fundamental data:
   - P/E Ratio
   - Market Cap
   - 52-week high/low
   - Revenue growth
   - Debt to equity ratio
   - Return on equity (ROE)
5. Sends all data to Claude API for AI analysis
6. Returns structured recommendation:
   - Short-term verdict (1-4 weeks): BUY/HOLD/SELL with reasoning
   - Long-term verdict (1-3 years): BUY/HOLD/SELL with reasoning
   - Key risks to watch
   - Price targets (if determinable)

## Phase 2 — Web App (app.py)
Flask web interface where users can:
- Enter any NSE/BSE stock symbol
- See beautiful charts and analysis
- Get AI recommendations

## Code Style
- Clean, readable Python
- Each analysis function is separate
- Print progress so user can see what's happening
- Clear error handling if stock symbol is wrong

## Commands
- Run CLI: python3.13 analyzer.py
- Run web: python3.13 app.py

## Important Disclaimer
All analysis is for educational purposes only — not financial advice.

## What NOT to do
- No real-time streaming data
- No user authentication yet
- Keep Phase 1 simple before adding web interface