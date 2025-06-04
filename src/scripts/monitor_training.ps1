# Monitor training progress in real-time
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptPath)
$logDir = Join-Path $projectRoot "tests"

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Show-Progress {
    param(
        [string]$Message,
        [double]$Value,
        [string]$Unit = "%",
        [string]$Color = "White"
    )
    $barLength = 30
    $filledLength = [math]::Min([math]::Max(0, [int]($Value * $barLength / 100)), $barLength)
    $bar = "[" + ("█" * $filledLength) + (" " * ($barLength - $filledLength)) + "]"
    Write-ColorOutput $Color "$Message $bar $Value$Unit"
}

function Show-Metrics {
    param(
        [hashtable]$Metrics
    )
    Clear-Host
    Write-ColorOutput "Cyan" "`n=== Minesweeper Training Monitor ==="
    Write-ColorOutput "Cyan" "Time: $(Get-Date -Format 'HH:mm:ss')"
    Write-ColorOutput "Cyan" "====================================`n"

    # Performance Metrics
    Write-ColorOutput "Yellow" "Performance Metrics:"
    $winRateColor = if ($Metrics['win_rate'] -ge 50) { "Green" } elseif ($Metrics['win_rate'] -ge 30) { "Yellow" } else { "Red" }
    Show-Progress "Win Rate" $Metrics['win_rate'] "%" $winRateColor
    
    $rewardColor = if ($Metrics['avg_reward'] -ge 2.0) { "Green" } elseif ($Metrics['avg_reward'] -ge 0.0) { "Yellow" } else { "Red" }
    Show-Progress "Average Reward" $Metrics['avg_reward'] "" $rewardColor
    
    Write-ColorOutput "White" "  Average Game Length: $($Metrics['avg_length']) steps`n"

    # Training Progress
    Write-ColorOutput "Yellow" "Training Progress:"
    Show-Progress "Timesteps" $Metrics['progress'] "%" "Cyan"
    Write-ColorOutput "White" "  Iterations: $($Metrics['iterations'])"
    Write-ColorOutput "White" "  Learning Phase: $($Metrics['learning_phase'])`n"

    # Recent Performance
    Write-ColorOutput "Yellow" "Recent Performance:"
    Write-ColorOutput "White" "  Last 5 Rewards: $($Metrics['recent_rewards'])"
    Write-ColorOutput "White" "  Last 5 Lengths: $($Metrics['recent_lengths'])`n"

    # Best Performance
    Write-ColorOutput "Yellow" "Best Performance:"
    Write-ColorOutput "Green" "  Best Win Rate: $($Metrics['best_win_rate'])%"
    Write-ColorOutput "Green" "  Best Reward: $($Metrics['best_reward'])`n"
}

# Initialize metrics
$metrics = @{
    'win_rate' = 0
    'avg_reward' = 0
    'avg_length' = 0
    'progress' = 0
    'iterations' = 0
    'learning_phase' = "Initial Random"
    'recent_rewards' = @()
    'recent_lengths' = @()
    'best_win_rate' = 0
    'best_reward' = 0
}

# Wait for the log file to exist
Write-ColorOutput "Yellow" "Waiting for training to start..."
$waitStartTime = Get-Date
$maxWaitTime = 300  # 5 minutes timeout

while ($true) {
    # Look for the most recent test_logs directory
    $testLogs = Get-ChildItem -Path $logDir -Directory -Filter "test_logs_*" | 
                Sort-Object LastWriteTime -Descending | 
                Select-Object -First 1

    if ($testLogs) {
        $logFile = Join-Path $testLogs.FullName "latest.log"
        if (Test-Path $logFile) {
            Write-ColorOutput "Green" "Found log file at: $logFile"
            break
        }
    }

    if ((Get-Date).Subtract($waitStartTime).TotalSeconds -gt $maxWaitTime) {
        Write-ColorOutput "Red" "`nError: Log file not found after 5 minutes of waiting."
        Write-ColorOutput "Yellow" "Please ensure the training script is running and creating the log file."
        Write-ColorOutput "Yellow" "`nPress any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }
    Write-ColorOutput "Yellow" "Waiting for log file... ($([math]::Floor((Get-Date).Subtract($waitStartTime).TotalSeconds)) seconds elapsed)"
    Start-Sleep -Seconds 5
}

Write-ColorOutput "Green" "Training log found. Starting monitoring..."
Write-ColorOutput "Yellow" "Press Ctrl+C to exit`n"

# Watch the log file for changes
try {
    Get-Content $logFile -Wait | ForEach-Object {
        $line = $_
        
        # Update metrics based on log content
        if ($line -match "Win Rate: ([\d.]+)%") {
            $metrics['win_rate'] = [double]$Matches[1]
            $metrics['best_win_rate'] = [math]::Max($metrics['best_win_rate'], $metrics['win_rate'])
        }
        if ($line -match "Average Reward: ([\d.-]+)") {
            $metrics['avg_reward'] = [double]$Matches[1]
            $metrics['best_reward'] = [math]::Max($metrics['best_reward'], $metrics['avg_reward'])
        }
        if ($line -match "Average Game Length: ([\d.]+)") {
            $metrics['avg_length'] = [double]$Matches[1]
        }
        if ($line -match "Progress: ([\d.]+)%") {
            $metrics['progress'] = [double]$Matches[1]
        }
        if ($line -match "Iteration (\d+)") {
            $metrics['iterations'] = [int]$Matches[1]
        }
        if ($line -match "Learning Phase: (.+)") {
            $metrics['learning_phase'] = $Matches[1]
        }
        if ($line -match "Last 10 Rewards: \[(.*?)\]") {
            $rewards = $Matches[1] -split ", " | ForEach-Object { [double]$_ }
            $metrics['recent_rewards'] = $rewards[-5..-1]
        }
        if ($line -match "Last 10 Game Lengths: \[(.*?)\]") {
            $lengths = $Matches[1] -split ", " | ForEach-Object { [double]$_ }
            $metrics['recent_lengths'] = $lengths[-5..-1]
        }

        # Update display
        Show-Metrics $metrics
    }
}
catch {
    Write-ColorOutput "Red" "`nMonitoring stopped."
    Write-ColorOutput "Yellow" "Final metrics:"
    Show-Metrics $metrics
    Write-ColorOutput "Yellow" "`nPress any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
