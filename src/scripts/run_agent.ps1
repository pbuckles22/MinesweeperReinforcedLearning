# Create training logs directory with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs\training_logs_$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Function to write to log file
function Write-ToLog {
    param(
        [string]$Message,
        [string]$LogFile = "$logDir\latest.log"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -FilePath $LogFile -Append
    Write-Host "$timestamp - $Message"
}

# Log start of training
Write-ToLog "Starting training run..."

# Training parameters
$boardSize = 8  # Standard board size
$maxMines = 10  # Standard mine count
$learningRate = 0.0003
$batchSize = 64
$nSteps = 2048
$nEpochs = 10
$gamma = 0.99
$gaeLambda = 0.95
$clipRange = 0.2
$entCoef = 0.01
$vfCoef = 0.5
$maxGradNorm = 0.5
$debugLevel = 2

# Select training mode
Write-Host "`nSelect training mode:"
Write-Host "1. Standard Training (100,000 timesteps)"
Write-Host "2. Extended Training (500,000 timesteps)"
Write-Host "3. Curriculum Learning (1,000,000 timesteps)"
$trainingMode = Read-Host "Enter mode (1-3)"

switch ($trainingMode) {
    "1" {
        $timesteps = 100000
        $trainingArgs = @(
            "src\core\train_agent.py",
            "--board-size", $boardSize,
            "--max-mines", $maxMines,
            "--timesteps", $timesteps,
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
            "--debug-level", $debugLevel
        )
    }
    "2" {
        $timesteps = 500000
        $trainingArgs = @(
            "src\core\train_agent.py",
            "--board-size", $boardSize,
            "--max-mines", $maxMines,
            "--timesteps", $timesteps,
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
            "--debug-level", $debugLevel
        )
    }
    "3" {
        $timesteps = 1000000
        $trainingArgs = @(
            "src\core\train_agent.py",
            "--board-size", $boardSize,
            "--max-mines", $maxMines,
            "--timesteps", $timesteps,
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
            "--curriculum"
        )
    }
    default {
        Write-Host "Invalid training mode selected. Exiting."
        exit 1
    }
}

# Start the training process
$process = Start-Process python -ArgumentList $trainingArgs -NoNewWindow -PassThru

# Monitor the training process
$startTime = Get-Date
$timeout = 7200  # 2 hours timeout
$checkInterval = 60  # Check every minute

while (-not $process.HasExited) {
    if ((Get-Date).Subtract($startTime).TotalSeconds -gt $timeout) {
        Write-ToLog "Training timeout reached. Stopping process..."
        Stop-Process -Id $process.Id -Force
        break
    }

    # Check for new log entries
    if (Test-Path "$logDir\latest.log") {
        $latestLog = Get-Content "$logDir\latest.log" -Tail 1
        Write-ToLog "Latest log: $latestLog"
    }

    Start-Sleep -Seconds $checkInterval
}

if ($process.HasExited) {
    Write-ToLog "Training completed with exit code: $($process.ExitCode)"
} else {
    Write-ToLog "Training was stopped due to timeout"
} 