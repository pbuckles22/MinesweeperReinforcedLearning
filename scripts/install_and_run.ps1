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

# Upgrade pip to latest version
Write-Host "Upgrading pip to latest version..."
python -m pip install --upgrade pip

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

# Permanently set PYTHONPATH in the virtual environment activation script
$activateScript = "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    # Add PYTHONPATH line at the very end of the activation script (before signature block)
    $pyPathLine = '$env:PYTHONPATH = "src;$env:PYTHONPATH"'
    
    # Check if PYTHONPATH is already set in the activation script
    $content = Get-Content $activateScript -Raw
    if ($content -notcontains $pyPathLine) {
        Write-Host "Setting PYTHONPATH in virtual environment activation script..."
        # Add the PYTHONPATH line at the end, before any signature block
        Add-Content -Path $activateScript -Value ""
        Add-Content -Path $activateScript -Value "# Set project PYTHONPATH"
        Add-Content -Path $activateScript -Value $pyPathLine
    }
}

# Create logs directory
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Run environment tests
Write-Host "Running environment tests..."
python -m pytest tests/integration/test_environment.py -v

# Run tests
Write-Host "Running tests..."
python -m pytest tests/unit/core tests/unit/agent tests/integration tests/functional tests/scripts -v

# Run training script
Write-Host "Starting training..."
python src/core/train_agent.py `
    --total_timesteps 1000000 `
    --learning_rate 0.0003 `
    --batch_size 64 `
    --n_steps 2048 `
    --n_epochs 10 `
    --gamma 0.99 `
    --gae_lambda 0.95 `
    --clip_range 0.2 `
    --ent_coef 0.01 `
    --vf_coef 0.5 `
    --max_grad_norm 0.5 `
    --eval_freq 10000 `
    --n_eval_episodes 100 `
    --save_freq 50000 `
    --verbose 1

# Deactivate virtual environment
deactivate 