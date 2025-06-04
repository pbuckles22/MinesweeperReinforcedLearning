param(
    [switch]$Force,
    [switch]$NoCache,
    [switch]$UseGPU
)

# Check if virtual environment exists
if (Test-Path "venv") {
    $response = Read-Host "Virtual environment already exists. Delete it? (y/n)"
    if ($response -ne "y") {
        exit 1
    }
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Recurse -Force venv
}

# Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r requirements.txt

# Install PyTorch with CUDA if requested
if ($UseGPU) {
    Write-Host "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

# Run environment tests
Write-Host "Running environment tests..."
python test_environment.py

# If tests pass, run training script
if ($LASTEXITCODE -eq 0) {
    Write-Host "Environment tests passed! Running training script..."
    python train_agent.py
} else {
    Write-Host "Environment tests failed. Please check the output above."
    exit 1
} 