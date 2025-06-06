param(
    [switch]$Force,
    [switch]$NoCache,
    [switch]$UseGPU
)

# Check if Python is installed
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed. Please install Python 3.8 or higher."
    exit 1
}

# Check Python version
$pythonVersion = python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ([version]$pythonVersion -lt [version]"3.8") {
    Write-Host "Python version $pythonVersion is not supported. Please install Python 3.8 or higher."
    exit 1
}

# Handle -Force parameter: Delete existing venv if it exists
if ($Force -and (Test-Path "venv")) {
    Write-Host "Force flag detected. Removing existing virtual environment..."
    Remove-Item -Path "venv" -Recurse -Force
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "venv\Scripts\Activate.ps1"

# Handle -NoCache parameter: Clear pip cache and install without using cache
if ($NoCache) {
    Write-Host "NoCache flag detected. Clearing pip cache and installing without cache..."
    pip cache purge
    pip install --no-cache-dir -r requirements.txt
} else {
    Write-Host "Installing requirements..."
    pip install -r requirements.txt
}

# Add src directory to Python path
$env:PYTHONPATH = "src;$env:PYTHONPATH"

# Create logs directory
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Run environment tests
Write-Host "Running environment tests..."
python tests/test_environment.py

# Run training script
Write-Host "Starting training..."
python src/core/train_agent.py `
    --board-size 8 `
    --max-mines 12 `
    --timesteps 1000000 `
    --learning-rate 0.0001 `
    --batch-size 64 `
    --n-steps 2048 `
    --n-epochs 10 `
    --gamma 0.99 `
    --gae-lambda 0.95 `
    --clip-range 0.2 `
    --ent-coef 0.01 `
    --vf-coef 0.5 `
    --max-grad-norm 0.5 `
    --debug-level 2

# Deactivate virtual environment
deactivate 