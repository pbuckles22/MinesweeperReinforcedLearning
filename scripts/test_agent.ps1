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
$logDir = "logs\agent_test_$timestamp"
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
Write-ToLog "Starting agent test..."

# Run agent tests
Write-Host "Running agent tests..."
python tests/test_train_agent.py

# Exit with the same code as the test script
exit $LASTEXITCODE 