"""Phase 0: pull one year of daily historical bars from Alpaca and save to data/.

Run from the project root with:

    python -m src.data.fetch_data

Uses the free IEX data feed (works with paper-trading keys). We request fully
*adjusted* bars (splits + dividends) so the price series doesn't show fake jumps
on a split day — feeding unadjusted prices into a backtest is a classic, silent
source of garbage results.
"""

import os
import sys

from dotenv import load_dotenv

# alpaca-py: the current SDK. You build a request *object* and pass it to the client,
# rather than calling a function with loose arguments (the old SDK's style).
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

# Make `config` importable whether you run this as a module or directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402


def get_client() -> StockHistoricalDataClient:
    """Build the data client from keys in .env. Fails loudly if keys are missing."""
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise SystemExit(
            "Missing Alpaca keys. Copy .env.example to .env and paste your paper-trading "
            "API key and secret. Get them free at https://app.alpaca.markets (Paper account)."
        )
    return StockHistoricalDataClient(api_key, secret_key)


def fetch_and_save() -> None:
    client = get_client()

    request = StockBarsRequest(
        symbol_or_symbols=config.SYMBOLS,
        timeframe=TimeFrame.Day,
        start=config.START_DATE,
        end=config.END_DATE,
        adjustment=Adjustment.ALL,  # split + dividend adjusted
        feed=DataFeed.IEX,          # free tier
    )

    print(f"Requesting daily bars for {config.SYMBOLS}")
    print(f"From {config.START_DATE.date()} to {config.END_DATE.date()}\n")

    bars = client.get_stock_bars(request)

    # .df is a pandas DataFrame with a (symbol, timestamp) multi-index.
    df = bars.df
    if df.empty:
        raise SystemExit("No data returned. Check your keys and that markets had data in this range.")

    # Save one tidy CSV per symbol so each is easy to load and inspect later.
    for symbol in config.SYMBOLS:
        if symbol not in df.index.get_level_values("symbol"):
            print(f"  WARNING: no data for {symbol} — skipping.")
            continue
        symbol_df = df.xs(symbol, level="symbol")
        out_path = config.DATA_DIR / f"{symbol}.csv"
        symbol_df.to_csv(out_path)
        print(f"  {symbol}: {len(symbol_df):>4} rows  "
              f"({symbol_df.index.min().date()} -> {symbol_df.index.max().date()})  "
              f"-> {out_path.relative_to(config.PROJECT_ROOT)}")

    # Sanity-check print: the head of the first symbol so you can eyeball the columns.
    first = config.SYMBOLS[0]
    print(f"\nSample ({first}):")
    print(df.xs(first, level="symbol").head())


if __name__ == "__main__":
    fetch_and_save()
