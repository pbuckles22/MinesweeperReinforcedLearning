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
    
    # First, try to deactivate if it's currently active
    if ($env:VIRTUAL_ENV) {
        Write-Host "Deactivating current virtual environment..."
        deactivate
    }
    
    # Wait a moment for processes to finish
    Start-Sleep -Seconds 2
    
    # Try to remove the virtual environment with better error handling
    try {
        # First try to remove with standard method
        Remove-Item -Path "venv" -Recurse -Force -ErrorAction Stop
        Write-Host "Successfully removed existing virtual environment."
    }
    catch {
        Write-Host "Warning: Could not remove virtual environment completely. Some files may be locked."
        Write-Host "This is normal on Windows. The script will continue with the existing environment."
        Write-Host "If you encounter issues, please:"
        Write-Host "1. Close any terminals/IDEs using this environment"
        Write-Host "2. Restart your terminal"
        Write-Host "3. Run the script again without -Force flag"
        Write-Host ""
        
        # Try to remove what we can
        try {
            Get-ChildItem -Path "venv" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
        }
        catch {
            Write-Host "Some files could not be removed. Continuing with existing environment..."
        }
    }
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
    # Read the current content
    $content = Get-Content $activateScript -Raw
    
    # Check if PYTHONPATH is already set in the activation script
    $pyPathLine = '$env:PYTHONPATH = "src;$env:PYTHONPATH"'
    
    if ($content -notcontains $pyPathLine) {
        Write-Host "Setting PYTHONPATH in virtual environment activation script..."
        
        # Find the end of the script (before signature block)
        $lines = Get-Content $activateScript
        $newLines = @()
        $inSignatureBlock = $false
        
        foreach ($line in $lines) {
            if ($line -match "^# SIG # Begin signature block") {
                $inSignatureBlock = $true
            }
            
            if (-not $inSignatureBlock) {
                $newLines += $line
            }
        }
        
        # Add PYTHONPATH lines before signature block
        $newLines += ""
        $newLines += "# Set project PYTHONPATH"
        $newLines += $pyPathLine
        
        # Write back the modified content
        $newLines | Set-Content $activateScript -Encoding UTF8
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

# Ask user if they want to run RL training test
Write-Host ""
Write-Host "=" * 60
Write-Host "All tests passed! 🎉"
Write-Host "=" * 60
Write-Host ""
Write-Host "Would you like to run a quick RL training test to verify early learning works?"
Write-Host "This will run a short training session (10,000 timesteps) to test the fixes."
Write-Host ""
$runRLTest = Read-Host "Run RL test? (y/n)"

if ($runRLTest -eq "y" -or $runRLTest -eq "Y" -or $runRLTest -eq "yes" -or $runRLTest -eq "Yes") {
    Write-Host ""
    Write-Host "Starting RL training test..."
    Write-Host "This will run for ~1-2 minutes to verify early learning works correctly."
    Write-Host ""
    
    # Run a short training test
    python src/core/train_agent.py `
        --total_timesteps 10000 `
        --eval_freq 2000 `
        --n_eval_episodes 20 `
        --learning_rate 0.0003 `
        --verbose 1
    
    Write-Host ""
    Write-Host "=" * 60
    Write-Host "RL training test completed!"
    Write-Host "Check the output above to verify early learning is working."
    Write-Host "=" * 60
} else {
    Write-Host ""
    Write-Host "Skipping RL training test."
    Write-Host ""
    Write-Host "To run the RL training test later:"
    Write-Host "1. Activate the virtual environment:"
    Write-Host "   .\venv\Scripts\Activate.ps1"
    Write-Host ""
    Write-Host "2. Run a quick test (10k timesteps, ~1-2 minutes):"
    Write-Host "   python src/core/train_agent.py --total_timesteps 10000 --eval_freq 2000 --n_eval_episodes 20 --verbose 1"
    Write-Host ""
    Write-Host "3. Or run a longer test (50k timesteps, ~5-10 minutes):"
    Write-Host "   python src/core/train_agent.py --total_timesteps 50000 --eval_freq 5000 --n_eval_episodes 50 --verbose 1"
    Write-Host ""
    Write-Host "4. Or run the full training (1M timesteps, ~1-2 hours):"
    Write-Host "   python src/core/train_agent.py --total_timesteps 1000000 --eval_freq 10000 --n_eval_episodes 100 --verbose 1"
    Write-Host ""
    Write-Host "The training will show progress including win rates and learning phases."
}

# Deactivate virtual environment
deactivate 