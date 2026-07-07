"""
Registers two Windows Task Scheduler tasks for the trading system.

Run once from an elevated (Administrator) PowerShell:
    cd trading-research
    .venv\\Scripts\\activate
    python scripts/setup_scheduler.py

Tasks created:
    TradingSystem-DailyRun    — 08:30 ET Mon-Fri
    TradingSystem-Intraday    — 09:25 ET Mon-Fri
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


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

    # Task definitions: (task_name, module_name, time_HH:MM)
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

    for task_name, module_name, start_time in tasks:
        # Build the command to run: cd to project root, then run python -m module
        # This ensures imports work correctly
        run_cmd = f"cmd /c \"cd /d {project_root} && {python_exe} -m {module_name}\""

        # Build schtasks command
        schtasks_cmd = [
            "schtasks",
            "/Create",
            f"/TN", task_name,
            f"/TR", run_cmd,
            "/SC", "WEEKLY",
            "/D", "MON,TUE,WED,THU,FRI",
            "/ST", start_time,
            "/SD", start_date,
            "/F",  # Force overwrite if exists
        ]

        print(f"Registering {task_name} ({start_time} Mon-Fri)...")
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
    print("=" * 70)

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
