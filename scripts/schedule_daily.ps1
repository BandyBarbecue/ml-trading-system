# schedule_daily.ps1 - Daily trading pipeline runner for Windows Task Scheduler
# Usage: .\scripts\schedule_daily.ps1 [-DryRun]
#
# Designed to be called by Windows Task Scheduler at 8:30 AM ET on trading days.
# Activates the project venv, runs the daily pipeline, and logs all output.

param(
    [switch]$DryRun
)

# Resolve absolute paths so the script works regardless of where Task Scheduler
# sets the working directory.
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
$LogDir      = Join-Path $ProjectRoot "data\logs"

# Create log directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogFile = Join-Path $LogDir "daily_run_$(Get-Date -Format 'yyyy-MM-dd').log"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp  $Message" | Tee-Object -FilePath $LogFile -Append
}

# Check dry-run first to avoid log file creation issues
if ($DryRun) {
    Write-Host "=== Daily run (DRY RUN) ==="
    Write-Host "Project root: $ProjectRoot"
    Write-Host "Dry run: True"
    Write-Host "Dry run mode - skipping actual pipeline execution."
    Write-Host "=== Dry run complete ==="
    exit 0
}

Write-Log "=== Daily run starting ==="
Write-Log "Project root: $ProjectRoot"
Write-Log "Dry run: $DryRun"

# Activate virtual environment
if (-not (Test-Path $VenvActivate)) {
    Write-Log "ERROR: venv not found at $VenvActivate - run: python -m venv .venv"
    exit 1
}

& $VenvActivate

# Change to project root so relative imports work
Set-Location $ProjectRoot

# Build the python command
$PythonArgs = @("-m", "src.execution.daily_run")

Write-Log "Running: python $($PythonArgs -join ' ')"

# Run the pipeline, capturing exit code
& python @PythonArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
$ExitCode = $LASTEXITCODE

Write-Log "=== Daily run finished: exit code $ExitCode ==="
exit $ExitCode
