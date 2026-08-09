"""modal_app.py — Stage 1: run the daily pipeline on Modal instead of the laptop.

Stage 1 moves ONLY daily_run. The intraday runner stays on the laptop until
Stage 2, so that the infrastructure change and the trading-logic change never
land in the same step.

Deploy:
    modal deploy modal_app.py         # registers the cron; runs without you

Test before deploying (runs once, right now, submits nothing):
    modal run modal_app.py::daily_dry_run

Watch a scheduled run afterwards:
    modal app logs trading-system
"""

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# tzdata is NOT optional. daily_run.py, intraday_runner.py, reconciliation.py
# and market_calendar.py all call ZoneInfo("America/New_York"), and three
# modules use the legacy "US/Eastern" alias. On a slim Debian image with no
# tzdata, zoneinfo raises ZoneInfoNotFoundError at import and the whole run
# dies before step 1. The pip package ships its own copy of the IANA database
# that zoneinfo falls back to, which is the most reliable fix.
image = (
    # 3.14 matches the laptop's interpreter. requirements-modal.txt pins every
    # dependency to the laptop's exact version -- the loose requirements.txt
    # resolved pandas 3.0.5 against the local 2.3.3, which would have changed
    # what the pipeline computes rather than just where it runs.
    modal.Image.debian_slim(python_version="3.14")
    .pip_install_from_requirements("requirements-modal.txt")
    .env(
        {
            # Behavioural parity with the laptop. There are 22 naive
            # datetime.now() / date.today() call sites; three of them decide
            # STORED DATA KEYS (equity_curve row dates, intraday parquet
            # paths, IC month buckets). A UTC container would silently shift
            # those by an hour and, either side of midnight, by a whole day.
            # Pinning the container to London means the migration changes
            # where the code runs and nothing about what it computes.
            "TZ": "Europe/London",
            # Redirects config.DATA_DIR to the mounted Volume. Requires the
            # one-line config.py change documented in the migration plan.
            "TRADING_DATA_DIR": "/data",
            "PYTHONUNBUFFERED": "1",
            # Six modules put a literal em-dash in the logging FORMAT string
            # (daily_run, intraday_runner, feature_matrix, fetch_fundamentals,
            # fetch_insider, sp500_history), emitted through a StreamHandler
            # that never sets an encoding. On Windows stderr is cp1252 and
            # encodes it fine; on Linux it follows the locale, and under a
            # C/POSIX locale that resolves to ascii and every single log line
            # raises UnicodeEncodeError. Debian bookworm's glibc ships C.UTF-8
            # so this currently works -- observed fine in the 2026-08-09 run --
            # but that is an unguaranteed default propping up the whole log
            # path, and forcing UTF-8 costs nothing.
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    # Ships config.py and the src/ package. Non-Python files are excluded by
    # default, so the local data/ directory is NOT baked into the image --
    # it lives in the Volume, which is the whole point.
    .add_local_python_source("config", "src")
)

app = modal.App("trading-system", image=image)

# All persistent state: prices, news, models, nlp_cache, ic_tracker,
# equity_curve, risk_state. Everything outside this path is wiped when the
# container exits.
volume = modal.Volume.from_name("trading-data", create_if_missing=True)

# ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY, FRED_API_KEY.
# config.py calls load_dotenv() with no .env present, which is a silent no-op;
# os.getenv then reads these injected values. No code change needed.
secrets = [modal.Secret.from_name("trading-secrets")]


def _setup_logging() -> None:
    """Log to stdout (captured by Modal) and to the Volume (matches today)."""
    import logging
    import config
    from src.execution.daily_run import _today

    log_dir = config.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_dir / f"daily_run_{_today().strftime('%Y-%m-%d')}.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def _run(dry_run: bool) -> dict:
    import logging

    _setup_logging()
    from src.execution.daily_run import run_daily

    try:
        result = run_daily(dry_run=dry_run)
    finally:
        # Flush the Volume even on failure -- a partial run still produced
        # fetched prices, news parquets and a paid-for NLP cache, and losing
        # those means re-paying Anthropic for headlines we already extracted.
        volume.commit()

    logging.getLogger(__name__).info(
        "Modal daily run: step=%s/7 orders=%s errors=%s",
        result["step"], len(result["orders"]), len(result["errors"]),
    )
    # Return a JSON-safe summary; pandas Series in the raw dict won't survive
    # serialisation back to the caller.
    return {
        "date": result["date"],
        "step": result["step"],
        "orders": len(result["orders"]),
        "errors": result["errors"],
        "risk_decisions": result["risk_decisions"],
        "halted": result.get("halted", False),
    }


# ---------------------------------------------------------------------------
# The scheduled job
# ---------------------------------------------------------------------------
# 08:30 America/New_York, Mon-Fri. Modal resolves the timezone itself, so the
# US and UK DST transitions -- which the Windows scheduler needed manual
# re-registration for twice a year -- are handled for us.
#
# retries is deliberately 0. This function submits real orders; a blind retry
# after a partial step 7 risks duplicate fills. The pipeline is already
# restart-safe by design (risk_state halt flag, and compute_trades sizes
# against CURRENT positions), but an automatic retry is not worth the tail
# risk on a system whose first rule is "risk before returns". Failures surface
# in the Modal dashboard; re-run by hand.
@app.function(
    schedule=modal.Cron("30 8 * * 1-5", timezone="America/New_York"),
    volumes={"/data": volume},
    secrets=secrets,
    # MEASURED, not guessed: a full local run on 2026-08-09 took 153 minutes,
    # of which Step 3 (the feature matrix) was 147. Fridays additionally run
    # reconciliation, which builds a SECOND feature matrix and then backtests
    # it -- so a Friday can plausibly approach 5 hours, and Modal's vCPUs are
    # not necessarily faster than a local XPS core. 6 hours is headroom, not
    # extravagance; an unused timeout costs nothing because billing is per
    # second of actual execution.
    timeout=6 * 60 * 60,
    # Step 3 is a single-threaded Python loop, so extra cores do not speed up
    # 96% of the runtime. The 2nd core only earns its keep on the 1st of the
    # month, when LightGBM (n_jobs=-1) retrains. Drop to cpu=1 to save ~40%.
    cpu=2,
    # Observed peak working set was 141 MB. An earlier draft of this file said
    # 8192, which would have billed ~4x the memory line for nothing.
    memory=2048,
    max_containers=1,      # never two pipelines against one Volume
    retries=0,
    # Forces a cold container per run, which fixes a silent staleness bug for
    # free. config.py evaluates END_DATE/START_DATE at IMPORT time (lines
    # 43-44), and fetch_prices.py does `from config import END_DATE` -- a
    # by-value bind. In a warm container reused across days, END_DATE would
    # still hold the day the container booted, so fetch_prices would keep
    # requesting a window ending in the past and prices would quietly stop
    # updating while the run still reported success. A fresh interpreter per
    # invocation re-evaluates both. Costs only a cold start, which a
    # once-a-day cron pays anyway.
    single_use_containers=True,
)
def daily():
    return _run(dry_run=False)


# ---------------------------------------------------------------------------
# Manual entrypoints (no schedule -- these only run when you invoke them)
# ---------------------------------------------------------------------------
@app.function(
    volumes={"/data": volume},
    secrets=secrets,
    timeout=6 * 60 * 60,
    cpu=2,
    memory=2048,
    max_containers=1,
    single_use_containers=True,
)
def daily_dry_run():
    """Full pipeline, no orders. Use this to validate the migration."""
    return _run(dry_run=True)


@app.function(volumes={"/data": volume}, secrets=secrets, timeout=900)
def parity_check() -> dict:
    """Prove the container matches the laptop BEFORE spending 2.5h on a dry run.

    Lives in THIS file on purpose. A previous version sat in a separate script
    that did `from modal_app import image` at module scope; Modal re-imports the
    entrypoint file inside the container, where modal_app is not on the path
    (the image ships only `config` and `src`), so every container died with
    ModuleNotFoundError and the call retried for hours. The entrypoint file is
    always mounted, so a check defined here needs no extra local source.
    """
    import os
    import sys
    from datetime import datetime
    from zoneinfo import ZoneInfo

    out = {"python": sys.version.split()[0]}

    for name in [
        "pandas", "numpy", "pyarrow", "yfinance", "requests", "lightgbm",
        "sklearn", "scipy", "optuna", "shap", "anthropic", "vectorbt", "lxml",
    ]:
        try:
            out[name] = getattr(__import__(name), "__version__", "(none)")
        except Exception as exc:
            out[name] = f"IMPORT FAILED: {type(exc).__name__}: {exc}"

    # tzdata is the most likely silent killer: slim Debian ships no IANA db.
    for tz in ("America/New_York", "US/Eastern", "Europe/London"):
        try:
            ZoneInfo(tz)
            out[f"tz:{tz}"] = "OK"
        except Exception as exc:
            out[f"tz:{tz}"] = f"FAILED: {type(exc).__name__}"

    out["TZ_env"] = os.environ.get("TZ", "<unset>")
    out["naive_now"] = datetime.now().isoformat(timespec="seconds")

    # config wiring: DATA_DIR must point at the Volume and see the real data.
    try:
        import config
        out["DATA_DIR"] = str(config.DATA_DIR)
        out["END_DATE"] = str(config.END_DATE)
        out["prices_on_volume"] = len(list(config.PRICES_DIR.glob("*")))
        out["tickers_file"] = (config.DATA_DIR / "sp500_tickers.csv").exists()
        out["equity_curve"] = (config.DATA_DIR / "equity_curve.parquet").exists()
    except Exception as exc:
        out["config"] = f"FAILED: {type(exc).__name__}: {exc}"

    # Import the heavy modules so any 3.14 / pandas incompatibility surfaces
    # in two minutes rather than two and a half hours.
    for mod in (
        "src.execution.daily_run",
        "src.features.feature_matrix",
        "src.data.fetch_prices",
        "src.execution.paper_trader",
        "src.ml.regime_classifier",
    ):
        try:
            __import__(mod)
            out[mod] = "import OK"
        except Exception as exc:
            out[mod] = f"FAILED: {type(exc).__name__}: {exc}"

    return out


@app.function(
    volumes={"/data": volume},
    secrets=secrets,
    timeout=6 * 60 * 60,
    cpu=2,
    memory=2048,
    max_containers=1,
    single_use_containers=True,
)
def validation_dry_run(as_of: str = "2026-08-07") -> dict:
    """Exercise the FULL pipeline on a chosen trading day. Manual only.

    Why this exists: the first Modal dry run landed on a Sunday, so the NYSE
    calendar gate returned at step 0/7 and validated nothing beyond the Volume
    mount. Waiting for the next session would mean deploying an unproven
    container, so this pins the clock to a real trading day instead.

    run_daily() takes no date argument and reads the clock through
    daily_run._today(), so that is the single point to override. dry_run is
    hard-coded True and there is no parameter to change it -- this function
    cannot submit an order regardless of how it is called. It is deliberately
    unscheduled: nothing invokes it but a human.
    """
    import logging
    from datetime import datetime
    from zoneinfo import ZoneInfo

    _setup_logging()
    from src.execution import daily_run as dr

    stamp = datetime.fromisoformat(as_of).replace(
        hour=8, minute=30, tzinfo=ZoneInfo("America/New_York")
    )
    dr._today = lambda: stamp  # noqa: E731 -- see docstring

    try:
        result = dr.run_daily(dry_run=True)
    finally:
        volume.commit()

    logging.getLogger(__name__).info(
        "Validation run (as_of=%s): step=%s/7 orders=%s errors=%s",
        as_of, result["step"], len(result["orders"]), len(result["errors"]),
    )
    return {
        "as_of": as_of,
        "date": result["date"],
        "step": result["step"],
        "orders": len(result["orders"]),
        "errors": result["errors"],
        "risk_decisions": result["risk_decisions"],
        "halted": result.get("halted", False),
    }


@app.function(volumes={"/data": volume}, timeout=600)
def inspect():
    """Confirm the Volume actually contains what we uploaded."""
    import config

    out = {}
    for name, path in [
        ("prices", config.PRICES_DIR),
        ("news", config.NEWS_DIR),
        ("models", config.MODELS_DIR),
        ("features", config.FEATURES_DIR),
    ]:
        out[name] = len(list(path.glob("*"))) if path.exists() else "MISSING"
    for f in ("risk_state.json", "equity_curve.parquet", "target_weights.json",
              "sp500_tickers.csv"):
        out[f] = (config.DATA_DIR / f).exists()
    out["DATA_DIR"] = str(config.DATA_DIR)
    return out


@app.local_entrypoint()
def check():
    """`modal run modal_app.py::check` -> print the parity report.

    A local entrypoint is needed to SEE the result: `modal run` on a plain
    app.function executes it but does not echo its return value.
    """
    out = parity_check.remote()
    print("\n" + "=" * 66)
    print("CONTAINER PARITY CHECK")
    print("=" * 66)
    for k, v in out.items():
        print(f"{k:>28} : {v}")
    print("=" * 66)


@app.local_entrypoint()
def main():
    """`modal run modal_app.py::main` -> verify the Volume, then dry-run.

    The target must be named explicitly: with more than one local entrypoint
    defined, a bare `modal run modal_app.py` errors rather than picking one.
    """
    print("Volume contents:", inspect.remote())
    print("Dry run result:", daily_dry_run.remote())
