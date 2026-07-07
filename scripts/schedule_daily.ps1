# schedule_daily.ps1 - Daily trading pipeline runner for Windows Task Scheduler
# Usage: .\scripts\schedule_daily.ps1 [-DryRun]
#
# Designed to be called by Windows Task Scheduler at 8:30 AM ET on trading days.
#
# Note: daily_run.py writes its own detailed logs directly to data/logs/ via its
# own FileHandler. This script adds start/end markers around that output.
# We do NOT use "2>&1 | Tee-Object" because that pipeline causes PowerShell 5.1
# to corrupt $LASTEXITCODE and would double-write to the log file.

param(
    [switch]$DryRun
)

# Resolve absolute paths so the script works regardless of where Task Scheduler
# sets the working directory.
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Python      = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir      = Join-Path $ProjectRoot "data\logs"

# Create log directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogFile = Join-Path $LogDir "daily_run_$(Get-Date -Format 'yyyy-MM-dd').log"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp  $Message" | Tee-Object -FilePath $LogFile -Append
}

if ($DryRun) {
    Write-Host "=== Daily run (DRY RUN) ==="
    Write-Host "Project root: $ProjectRoot"
    Write-Host "Dry run mode - skipping actual pipeline execution."
    Write-Host "=== Dry run complete ==="
    exit 0
}

if (-not (Test-Path $Python)) {
    Write-Log "ERROR: Python not found at $Python - run: python -m venv .venv"
    exit 1
}

Write-Log "=== Daily run starting ==="
Write-Log "Project root: $ProjectRoot"
Write-Log "Python: $Python"

# Run from project root so relative imports (src.*) resolve correctly.
Set-Location $ProjectRoot

# PYTHONUNBUFFERED=1 ensures Python flushes log output immediately.
$env:PYTHONUNBUFFERED = "1"

$PythonArgs = @("-m", "src.execution.daily_run")

& $Python @PythonArgs
$ExitCode = $LASTEXITCODE

Write-Log "=== Daily run finished: exit code $ExitCode ==="
exit $ExitCode
