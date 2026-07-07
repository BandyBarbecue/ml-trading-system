"""
Removes Windows Task Scheduler tasks for the trading system.

Run from an elevated (Administrator) PowerShell:
    cd trading-research
    python scripts/remove_scheduler.py

Tasks removed:
    TradingSystem-DailyRun
    TradingSystem-Intraday
"""

import subprocess
import sys


def main() -> None:
    """Remove Task Scheduler tasks."""

    # Parse --check flag
    check_mode = "--check" in sys.argv

    # Task names to remove
    tasks = [
        "TradingSystem-DailyRun",
        "TradingSystem-Intraday",
    ]

    print("=" * 70)
    print(f"Trading System Task Scheduler Removal")
    print(f"Mode: {'DRY RUN' if check_mode else 'REMOVE'}")
    print("=" * 70)
    print()

    all_success = True

    for task_name in tasks:
        # Build schtasks command to delete
        schtasks_cmd = [
            "schtasks",
            "/Delete",
            f"/TN", task_name,
            "/F",  # Force delete without confirmation
        ]

        print(f"Removing {task_name}...")
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
        print("Dry-run complete. No tasks were removed.")
    elif all_success:
        print("Tasks removed successfully!")
    else:
        print("One or more tasks failed to remove.")

    print()
    print("Verify tasks are removed by running:")
    print('  schtasks /Query /TN "TradingSystem-DailyRun"')
    print('  schtasks /Query /TN "TradingSystem-Intraday"')
    print("=" * 70)

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
