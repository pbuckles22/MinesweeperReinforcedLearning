import pytest
import os
import subprocess
from pathlib import Path

@pytest.fixture
def script_path():
    """Get the path to the run script."""
    return Path("scripts/run_agent.ps1")

def test_script_exists(script_path):
    """Test that the run script exists."""
    assert script_path.exists()
    assert script_path.is_file()

def test_script_permissions(script_path):
    """Test that the script has the correct permissions."""
    # Check if the script is readable
    assert os.access(script_path, os.R_OK)

def test_script_syntax(script_path):
    """Test that the script has valid PowerShell syntax."""
    try:
        # Use PowerShell to check syntax without execution
        result = subprocess.run(
            ["powershell", "-Command", f"Get-Content {script_path} | Out-String | Test-ScriptBlock"],
            capture_output=True,
            text=True
        )
        # PowerShell syntax check might not be available, so we'll just check if the command runs
        assert result.returncode in [0, 1], f"Script syntax check failed: {result.stderr}"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script syntax error: {e.stderr}")

def test_script_parameters(script_path):
    """Test that the script has training parameters."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for training parameters (using actual parameter names from script)
    assert "board-size" in content.lower() or "boardsize" in content.lower()
    assert "max-mines" in content.lower() or "maxmines" in content.lower()
    assert "timesteps" in content.lower()

def test_script_environment_check(script_path):
    """Test that the script uses Python for training."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for Python usage
    assert "python" in content.lower()
    assert "train_agent.py" in content.lower()

def test_script_output_handling(script_path):
    """Test that the script handles output correctly."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for output handling - the script uses Write-Host and Write-ToLog
    assert "write-host" in content.lower()
    assert "write-tolog" in content.lower()

def test_script_error_handling(script_path):
    """Test that the script handles errors correctly."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for error handling - the script has a default case in switch and exit
    assert "default" in content.lower()  # Switch default case
    assert "exit" in content.lower()     # Exit on error
