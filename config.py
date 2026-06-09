"""Central configuration. Single source of truth for the universe, date range, and paths.

Importing from here (instead of hardcoding values in each script) means that when you
change the symbol list or the lookback window, you change it in exactly one place.
"""

from datetime import datetime, timedelta
from pathlib import Path

# --- Universe ---------------------------------------------------------------
# Five liquid names to start. Liquidity matters: thinly-traded stocks have wide
# spreads that make backtests lie to you about how easy it is to get filled.
SYMBOLS: list[str] = ["AAPL", "MSFT", "JNJ", "KO", "SPY"]

# --- Date range -------------------------------------------------------------
LOOKBACK_DAYS: int = 1825  # 5 years
END_DATE: datetime = datetime.now()
START_DATE: datetime = END_DATE - timedelta(days=LOOKBACK_DAYS)

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
