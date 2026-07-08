"""
Registers two Windows Task Scheduler tasks for the trading system.

Run once from an elevated (Administrator) PowerShell:
    cd trading-research
    .venv\\Scripts\\activate
    python scripts/setup_scheduler.py

Tasks created:
    TradingSystem-DailyRun    — 08:30 ET Mon-Fri
    TradingSystem-Intraday    — 09:25 ET Mon-Fri

This is the CANONICAL way to register the tasks. (An older create_task.ps1
was removed: it registered "08:30" as London wall-clock time — i.e. 03:30 ET —
and only covered the DailyRun task. This script converts ET to London time
explicitly; see _et_to_london_clock.)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

_ET     = ZoneInfo("America/New_York")
_LONDON = ZoneInfo("Europe/London")


def _et_to_london_clock(hh_mm: str) -> str:
    """Convert an ET wall-clock time (e.g. "08:30") to the equivalent
    Europe/London wall-clock time, computed for TODAY's date.

    Why this is needed: schtasks' /ST parameter is always interpreted in the
    MACHINE's local time zone. This box runs in Europe/London, but the
    trading schedule (market open, etc.) is defined in America/New_York.
    Previously this script passed the ET time string straight to /ST, which
    silently registered the task to fire at that clock time in LONDON —
    e.g. "08:30" meant 08:30 London time (03:30 or 08:30 ET depending on the
    season), not 08:30 ET as documented. That's up to a 5-hour miss.

    WARNING — DST misalignment weeks: the US and UK do not change clocks on
    the same calendar dates (they're offset by one to a few weeks each
    spring and autumn — e.g. the US typically springs forward about a week
    before the UK, and falls back about a week after). schtasks' /ST is a
    STATIC local time with no ongoing timezone awareness: once registered,
    it does NOT recompute itself as the US/UK DST calendars drift in and out
    of alignment. During those one-to-few-week mismatch windows each spring
    and autumn, a task registered on an ordinary day will fire up to one
    hour off from the intended ET time until it is re-registered. Re-run
    this script (or create_task.ps1) shortly after each DST changeover to
    keep the scheduled time accurate.
    """
    hour, minute = map(int, hh_mm.split(":"))
    today = datetime.now(_ET).date()
    et_dt = datetime(today.year, today.month, today.day, hour, minute, tzinfo=_ET)
    london_dt = et_dt.astimezone(_LONDON)
    return london_dt.strftime("%H:%M")


def main() -> None:
    """Register Task Scheduler tasks for daily run and intraday runner."""

    # Parse --check flag
    check_mode = "--check" in sys.argv

    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.resolve()
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"

    # Verify python exists
    if not python_exe.exists():
        print(f"ERROR: Python executable not found at {python_exe}")
        print("Make sure venv is set up: python -m venv .venv")
        sys.exit(1)

    # Start date is today — use DD/MM/YYYY to match UK locale Windows expects
    start_date = datetime.now().strftime("%d/%m/%Y")

    # Task definitions: (task_name, module_name, time_HH:MM in ET)
    tasks = [
        ("TradingSystem-DailyRun", "src.execution.daily_run", "08:30"),
        ("TradingSystem-Intraday", "src.execution.intraday_runner", "09:25"),
    ]

    print("=" * 70)
    print(f"Trading System Task Scheduler Setup")
    print(f"Project root: {project_root}")
    print(f"Python exe: {python_exe}")
    print(f"Start date: {start_date}")
    print(f"Mode: {'DRY RUN' if check_mode else 'REGISTER'}")
    print("=" * 70)
    print()

    all_success = True

    for task_name, module_name, et_time in tasks:
        london_time = _et_to_london_clock(et_time)

        # Build the command to run: cd to project root, then run python -m module.
        # Both the project root and the python.exe path are individually
        # double-quoted because this repo's path contains spaces
        # ("C:\Claude Projects SAFE\..."). Previously these were unquoted,
        # so `cd /d C:\Claude Projects SAFE\...` only ever cd'd into
        # "C:\Claude" (stopping at the first space) and the rest of the line
        # was misparsed — the task registered "successfully" but silently
        # failed on every single fire.
        #
        # IMPORTANT: do NOT additionally wrap this whole string in one more
        # outer pair of quotes (e.g. `cmd /c "..."`). cmd.exe's argument
        # parser toggles "inside quotes" mode on every quote character it
        # sees — it does not understand nesting. Adding an extra outer quote
        # pair around these two ALREADY-quoted paths shifts every quote's
        # parity, which flips the paths' quoted ranges to UNquoted and
        # reintroduces the exact "path breaks at the first space" bug this
        # fix is for. Individually quoting each path (with no outer wrapper)
        # is the standard, correct form — identical to typing
        # `cd /d "C:\Path With Spaces" && "C:\Path\python.exe" -m module`
        # directly at a cmd prompt.
        run_cmd = f'cmd /c cd /d "{project_root}" && "{python_exe}" -m {module_name}'

        # Build schtasks command
        schtasks_cmd = [
            "schtasks",
            "/Create",
            f"/TN", task_name,
            f"/TR", run_cmd,
            "/SC", "WEEKLY",
            "/D", "MON,TUE,WED,THU,FRI",
            "/ST", london_time,   # local (London) clock time equivalent to et_time — see _et_to_london_clock
            "/SD", start_date,
            "/F",  # Force overwrite if exists
        ]

        print(f"Registering {task_name} ({et_time} ET -> {london_time} London Mon-Fri)...")
        print(f"  Command: {' '.join(schtasks_cmd)}")

        if check_mode:
            print(f"  [DRY RUN] Not executed")
            print()
        else:
            try:
                result = subprocess.run(
                    schtasks_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode == 0:
                    print(f"  OK")
                else:
                    print(f"  FAILED (exit code {result.returncode})")
                    if result.stderr:
                        print(f"  Error: {result.stderr}")
                    all_success = False

                print()

            except Exception as e:
                print(f"  EXCEPTION: {e}")
                all_success = False
                print()

    # Summary
    print("=" * 70)
    if check_mode:
        print("Dry-run complete. No tasks were registered.")
    elif all_success:
        print("Tasks registered successfully!")
    else:
        print("One or more tasks failed to register.")

    print()
    print("Verify in Task Scheduler (taskschd.msc) or run:")
    print('  schtasks /Query /TN "TradingSystem-DailyRun"')
    print('  schtasks /Query /TN "TradingSystem-Intraday"')
    print()
    print("Reminder: the /ST times above are London wall-clock times computed")
    print("today to match the documented ET schedule. Re-run this script near")
    print("US/UK DST changeovers to keep them aligned (see _et_to_london_clock).")
    print("=" * 70)

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
