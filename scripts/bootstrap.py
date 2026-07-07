"""
One-time setup script. Run this once before the automated system takes over.

What it does (in order):
  1. Downloads 5 years of daily prices for all S&P 500 tickers  (~30-60 min)
  2. Downloads VIX history                                        (~5 sec)
  3. Downloads earnings surprise history (needs FMP_API_KEY)      (~5 min)
  4. Trains the first LightGBM model and saves it to disk         (~5-10 min)

After this script completes, the Task Scheduler jobs handle everything
automatically. You never need to run this again unless you wipe the data folder.

Usage:
    cd trading-research
    .venv\\Scripts\\activate
    python scripts/bootstrap.py
"""

import sys
import time
from pathlib import Path

# Make sure imports resolve from the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def step(n: int, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {n}: {label}")
    print(f"{'='*60}")


def main() -> None:
    print("\nTrading System Bootstrap")
    print("This will take 30-60 minutes on first run.\n")

    # ------------------------------------------------------------------
    # Step 1: Prices
    # ------------------------------------------------------------------
    step(1, "Downloading 5 years of daily prices (S&P 500)")
    print("This is the slow one — ~500 tickers from Alpaca.")
    print("It saves progress as it goes; safe to Ctrl+C and re-run.\n")
    try:
        from src.data.fetch_prices import update_prices
        update_prices()
        print("\n✓ Prices done.")
    except Exception as exc:
        print(f"\n✗ Prices failed: {exc}")
        print("Check your ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: VIX
    # ------------------------------------------------------------------
    step(2, "Downloading VIX history")
    try:
        from src.data.fetch_vix import update_vix
        update_vix()
        print("✓ VIX done.")
    except Exception as exc:
        print(f"✗ VIX failed: {exc}")
        print("  (non-fatal — VIX regime filter will default to 20.0)")

    # ------------------------------------------------------------------
    # Step 3: Earnings
    # ------------------------------------------------------------------
    step(3, "Downloading earnings surprises (FMP API)")
    import os
    if not os.getenv("FMP_API_KEY"):
        print("Skipping — FMP_API_KEY not set in .env.")
        print("Add it later; the earnings (SUE) feature will be NaN until then.")
    else:
        try:
            from src.data.fetch_earnings import update_earnings
            update_earnings()
            print("✓ Earnings done.")
        except Exception as exc:
            print(f"✗ Earnings failed: {exc}")
            print("  (non-fatal — SUE feature will be NaN)")

    # ------------------------------------------------------------------
    # Step 4: Train the first model
    # ------------------------------------------------------------------
    step(4, "Training the first LightGBM model")
    print("Builds the feature matrix then trains on the full history.")
    print("This takes 5-10 minutes.\n")
    try:
        import pandas as pd
        import config
        from src.data.sp500_universe import get_symbols
        from src.features.feature_matrix import build_feature_matrix
        from src.ml.model import train

        symbols = get_symbols()
        print(f"Universe: {len(symbols)} tickers")

        dates = (
            pd.bdate_range(end=pd.Timestamp.today(), periods=config.MODEL_TRAIN_DAYS + 25)
            .strftime("%Y-%m-%d")
            .tolist()
        )
        print(f"Building feature matrix for {len(dates)} dates...")
        feature_matrix = build_feature_matrix(symbols, dates)
        print(f"Feature matrix: {len(feature_matrix)} rows")

        print("Training LightGBM model (this is the slow part)...")
        model = train(feature_matrix, save=True)
        print("✓ Model trained and saved to data/models/")
    except Exception as exc:
        print(f"✗ Model training failed: {exc}")
        print("  The system will retry on the 1st of next month automatically.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Bootstrap complete.")
    print(f"{'='*60}")
    print("""
What happens next (automatically, no action needed):

  Every weekday at 08:30  — fetches today's data, runs ML model,
                            rebalances portfolio on Mondays, submits
                            paper orders to Alpaca.

  Every weekday at 09:25  — intraday loop runs until 15:50,
                            fires entry timing and event-driven trades.

  1st of each month       — model retrains on fresh 2-year window,
                            Optuna tunes hyperparameters, SHAP report
                            saved to data/attribution/.

Monitor your paper positions at: https://app.alpaca.markets
""")


if __name__ == "__main__":
    main()
