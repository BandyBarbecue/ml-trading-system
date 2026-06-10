# create_task.ps1 — Register the daily trading task with Windows Task Scheduler
#
# Run this ONCE as Administrator to set up the scheduled task.
# After running, you can view/modify the task in Task Scheduler (taskschd.msc).
#
# The task runs at 08:30 AM daily (Monday-Friday).
# Adjust the trigger time if you are in a different timezone.

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$RunScript   = Join-Path $ScriptDir "schedule_daily.ps1"
$TaskName    = "TradingAlgorithm_DailyRun"

# Unregister existing task if it exists (clean re-registration)
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

$action  = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File `"$RunScript`"" `
    -WorkingDirectory $ProjectRoot

# Mon-Fri at 08:30 — adjust timezone offset if needed
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "08:30AM"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action   $action `
    -Trigger  $trigger `
    -Settings $settings `
    -Description "Run the trading algorithm daily pipeline at 8:30 AM" `
    -RunLevel Highest

Write-Host "Task '$TaskName' registered successfully."
Write-Host "View/edit in Task Scheduler: taskschd.msc"
