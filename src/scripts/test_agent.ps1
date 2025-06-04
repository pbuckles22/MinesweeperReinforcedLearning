# Activate virtual environment if it exists
if (Test-Path "..\..\venv\Scripts\Activate.ps1") {
    & "..\..\venv\Scripts\Activate.ps1"
} else {
    Write-Host "Virtual environment not found. Please create it first."
    exit 1
}

# Set working directory to project root
Set-Location -Path "..\.."

# Add src directory to Python path
$env:PYTHONPATH = "src;$env:PYTHONPATH"

# Run environment tests first
Write-Host "Running environment tests..."
python "src\tests\test_environment.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment tests failed. Please fix the issues before proceeding."
    exit 1
}

# Create test logs directory with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "tests\test_logs_$timestamp"
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
Write-ToLog "Starting training..."

# Training parameters
$boardSize = 10
$maxMines = 10
$timesteps = 1000000
$learningRate = 0.0003
$batchSize = 64
$bufferSize = 100000
$gamma = 0.99
$tau = 0.005
$updateFrequency = 4
$checkpointFrequency = 10000
$evalFrequency = 1000
$evalEpisodes = 10
$saveDir = "tests\test_models_$timestamp"

# Create save directory
New-Item -ItemType Directory -Force -Path $saveDir | Out-Null

# Start training
$trainingArgs = @(
    "src\core\train_agent.py",
    "--board-size", $boardSize,
    "--max-mines", $maxMines,
    "--timesteps", $timesteps,
    "--learning-rate", $learningRate,
    "--batch-size", $batchSize,
    "--buffer-size", $bufferSize,
    "--gamma", $gamma,
    "--tau", $tau,
    "--update-frequency", $updateFrequency,
    "--checkpoint-frequency", $checkpointFrequency,
    "--eval-frequency", $evalFrequency,
    "--eval-episodes", $evalEpisodes,
    "--save-dir", $saveDir,
    "--log-dir", $logDir
)

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