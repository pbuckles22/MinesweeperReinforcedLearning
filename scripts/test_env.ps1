# Activate virtual environment if it exists
if (Test-Path "venv") {
    Write-Host "Activating virtual environment..."
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Error: Virtual environment not found. Please run install_and_run.ps1 first."
    exit 1
}

# Check if requirements.txt exists
if (Test-Path "requirements.txt") {
    Write-Host "Installing/updating dependencies..."
    pip install -r requirements.txt
} else {
    Write-Host "Error: requirements.txt not found."
    exit 1
}

# Run tests
Write-Host "Running environment tests..."
python -m pytest tests/integration/test_environment.py -v

# Exit with the same code as the test script
exit $LASTEXITCODE 