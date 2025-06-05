# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Virtual environment not found. Please create it first."
    exit 1
}

# Add src directory to Python path
$env:PYTHONPATH = "src;$env:PYTHONPATH"

# Set console output encoding to UTF-8 to handle emojis
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Create test logs directory with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs\early_learning_test_$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Function to write to log file
function Write-ToLog {
    param(
        [string]$Message,
        [string]$LogFile = "$logDir\latest.log"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -FilePath $LogFile -Append -Encoding UTF8
    Write-Host "$timestamp - $Message"
}

# Log start of test
Write-ToLog "Starting early learning test..."

# Test parameters - focused on early learning
$boardSize = 4  # Start with 4x4
$maxMines = 2   # Start with 2 mines
$learningRate = 0.0003  # Higher learning rate for faster learning
$batchSize = 32  # Smaller batch size for more frequent updates
$nSteps = 1024  # Fewer steps per update
$nEpochs = 10
$gamma = 0.99
$gaeLambda = 0.95
$clipRange = 0.2
$entCoef = 0.02  # Higher entropy for more exploration
$vfCoef = 0.5
$maxGradNorm = 0.5
$debugLevel = 2
$randomSeed = 42

# Test parameters
$totalTimesteps = 10000  # Much shorter for quick testing
$targetWinRate = 0.3  # Target 30% win rate for early learning
$timeout = 300  # 5 minutes timeout

# Network architecture - simpler for early learning
$policyKwargs = @{
    net_arch = @(
        @{
            pi = @(64, 64)  # Smaller network for early learning
            vf = @(64, 64)
        }
    )
}

function Start-Training {
    param(
        [int]$Timesteps
    )
    
    Write-ToLog "Starting early learning test with $Timesteps timesteps..."
    Write-ToLog "Target: $($targetWinRate * 100)% win rate on ${boardSize}x${boardSize} board with $maxMines mines"
    Write-ToLog "This test focuses on early learning phase only"
    
    # Training arguments for early learning test
    $trainingArgs = @(
        "src/core/train_agent.py",
        "--board-size", $boardSize,
        "--max-mines", $maxMines,
        "--timesteps", $Timesteps,
        "--learning-rate", $learningRate,
        "--batch-size", $batchSize,
        "--n-steps", $nSteps,
        "--n-epochs", $nEpochs,
        "--gamma", $gamma,
        "--gae-lambda", $gaeLambda,
        "--clip-range", $clipRange,
        "--ent-coef", $entCoef,
        "--vf-coef", $vfCoef,
        "--max-grad-norm", $maxGradNorm,
        "--debug-level", $debugLevel,
        "--random-seed", $randomSeed
    )

    # Start the training process
    $process = Start-Process python -ArgumentList $trainingArgs -NoNewWindow -PassThru

    # Monitor the training process
    $startTime = Get-Date
    $checkInterval = 10  # Check every 10 seconds
    $lastWinRate = 0
    $stagnantCount = 0

    while (-not $process.HasExited) {
        if ((Get-Date).Subtract($startTime).TotalSeconds -gt $timeout) {
            Write-ToLog "Test timeout reached. Stopping process..."
            Stop-Process -Id $process.Id -Force
            return $false
        }

        # Check for new log entries
        if (Test-Path "$logDir\latest.log") {
            $latestLog = Get-Content "$logDir\latest.log" -Tail 1 -Encoding UTF8
            Write-ToLog "Latest log: $latestLog"
            
            # Check for evaluation results
            if ($latestLog -match "Evaluation Results:") {
                $evalResults = @()
                for ($i = 1; $i -le 5; $i++) {
                    $nextLine = Get-Content "$logDir\latest.log" -Tail ($i + 1) -Head 1 -Encoding UTF8
                    $evalResults += $nextLine
                }
                Write-ToLog "Evaluation Results:`n$($evalResults -join "`n")"
                
                # Extract win rate from evaluation results
                if ($evalResults[0] -match "Win Rate: (\d+\.\d+)%") {
                    $currentEvalWinRate = [double]$Matches[1] / 100
                    Write-ToLog "Current evaluation win rate: $($currentEvalWinRate * 100)%"
                    
                    # Check for stagnation
                    if ([Math]::Abs($currentEvalWinRate - $lastWinRate) -lt 0.01) {
                        $stagnantCount++
                        if ($stagnantCount -ge 3) {
                            Write-ToLog "Training appears to have stagnated. Current win rate: $($currentEvalWinRate * 100)%"
                            Stop-Process -Id $process.Id -Force
                            return $currentEvalWinRate -ge $targetWinRate
                        }
                    } else {
                        $stagnantCount = 0
                    }
                    $lastWinRate = $currentEvalWinRate
                    
                    if ($currentEvalWinRate -ge $targetWinRate) {
                        Write-ToLog "Target win rate of $($targetWinRate * 100)% achieved!"
                        Stop-Process -Id $process.Id -Force
                        return $true
                    }
                }
            }
        }

        Start-Sleep -Seconds $checkInterval
    }

    return $false
}

# Run the early learning test
Write-ToLog "Starting early learning test with $totalTimesteps timesteps"
Write-ToLog "Board: ${boardSize}x${boardSize} with $maxMines mines (${([math]::Round($maxMines/($boardSize*$boardSize)*100,1))}% density)"
Write-ToLog "This test will run for up to 5 minutes to validate the early learning phase"
$success = Start-Training -Timesteps $totalTimesteps

if ($success) {
    Write-ToLog "Early learning test passed with win rate >= $($targetWinRate * 100)%"
    Write-ToLog "The agent has demonstrated the ability to learn basic mine avoidance and safe exploration"
} else {
    Write-ToLog "Early learning test failed - did not achieve target win rate of $($targetWinRate * 100)%"
    Write-ToLog "This indicates issues with the early learning phase that need to be addressed"
} 