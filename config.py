# config.py — single source of truth for the entire system
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR      = PROJECT_ROOT / "data"

# --- Data directories ---
PRICES_DIR    = DATA_DIR / "prices"      # one parquet per symbol
INTRADAY_DIR  = DATA_DIR / "intraday"   # ticker/YYYY-MM-DD.parquet
NEWS_DIR      = DATA_DIR / "news"        # YYYY-MM-DD.parquet
EARNINGS_FILE = DATA_DIR / "earnings" / "earnings.parquet"
FUNDAMENTALS_FILE = DATA_DIR / "fundamentals" / "fundamentals.parquet"
INSIDER_FILE      = DATA_DIR / "insider"       / "insider.parquet"
REGIME_FILE       = DATA_DIR / "features"      / "regime.parquet"
# NOTE: the St. Louis Fed is a Federal Reserve BANK, not a government agency —
# its domain is stlouisfed.ORG. The previous ".gov" hostname never existed
# (DNS NXDOMAIN), so every macro fetch had silently failed since day one.
FRED_BASE_URL     = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")

# Long-short extension — disabled by default until backtest confirms positive IC
LONG_SHORT_ENABLED  = False
N_SHORT_POSITIONS   = 25
MAX_SHORT_PCT       = 0.03   # 3% max per short position
GROSS_EXPOSURE_CAP  = 1.60   # long + |short| <= 160% of NAV
FEATURES_DIR  = DATA_DIR / "features"
MODELS_DIR    = DATA_DIR / "models"
ATTR_DIR      = DATA_DIR / "attribution"
UNIVERSE_FILE = DATA_DIR / "sp500_tickers.csv"
VIX_FILE      = DATA_DIR / "vix.parquet"

# --- Universe ---
# Populated from UNIVERSE_FILE at runtime; this is the fallback for quick tests
FALLBACK_SYMBOLS = ["AAPL", "MSFT", "JNJ", "KO", "SPY"]
SYMBOLS = FALLBACK_SYMBOLS  # legacy alias — small fixed universe used by the pre-ML modules

# --- Date range (5-year lookback) ---
END_DATE   = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")

# --- Model ---
MODEL_TRAIN_DAYS  = 504   # 2-year rolling training window
TARGET_HORIZON    = 20    # predict 20-day forward return rank
IC_WINDOW         = 60    # rolling IC computation window (trading days)
IC_RETIRE_MONTHS  = 3     # flag feature if IC < threshold for this many months
IC_RETIRE_THRESH  = 0.01  # IC below this is "near-zero"
OPTUNA_TRIALS     = 50    # hyperparameter search trials

# --- Portfolio ---
N_POSITIONS       = 25
MAX_POSITION_PCT  = 0.08   # 8% max per stock
MAX_SECTOR_PCT    = 0.30   # 30% max per GICS sector
REBALANCE_WEEKDAY = 0      # Monday (0=Mon … 4=Fri)
TURNOVER_CAP      = 0.40   # max 40% portfolio turnover per rebalance
HRP_LOOKBACK      = 60     # days of returns for HRP covariance

# --- Risk ---
DAILY_LOSS_LIMIT      = 0.02   # 2% intraday drop → exit all
MAX_DRAWDOWN          = 0.15   # 15% from peak → scale to 50%
DRAWDOWN_SCALE_FACTOR = 0.50   # multiplier applied on drawdown breach (kept separate from
                                # VIX_SCALE_FACTOR so the two controls can be tuned independently)
VIX_REGIME_THRESHOLD  = 30     # VIX > 30 → halve positions
VIX_SCALE_FACTOR      = 0.50

# --- Intraday ---
INTRADAY_BAR_MINUTES  = 30
ENTRY_WINDOW_BARS     = 3      # wait up to 3 × 30-min bars (90 min)
EVENT_CONFIDENCE_MIN  = 0.85   # Claude confidence threshold for event trades
EVENT_POSITION_MAX    = 0.02   # max 2% per event-driven trade
INTRADAY_CLOSE_TIME   = "15:50"

# --- Claude API ---
CLAUDE_NLP_MODEL    = "claude-haiku-4-5-20251001"
CLAUDE_REVIEW_MODEL = "claude-sonnet-4-6"
NLP_BATCH_SIZE      = 20      # headlines per Claude API call
NLP_DECAY_DAYS      = 3       # recency decay window

# --- FMP API ---
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_API_KEY  = os.getenv("FMP_API_KEY", "")

# --- Alpaca ---
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER      = True   # always paper, never live
