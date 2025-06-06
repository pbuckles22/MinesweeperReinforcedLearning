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
    # Check if the script is executable
    assert os.access(script_path, os.X_OK)

def test_script_syntax(script_path):
    """Test that the script has valid PowerShell syntax."""
    try:
        # Use PowerShell to check syntax without execution
        result = subprocess.run(
            ["powershell", "-Command", f"Get-Content {script_path} | Out-String | Test-ScriptBlock"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Script syntax error: {result.stderr}"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script syntax error: {e.stderr}")

def test_script_parameters(script_path):
    """Test that the script accepts required parameters."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for parameter definitions
    assert "param" in content.lower()
    assert "board_size" in content.lower()
    assert "mines" in content.lower()

def test_script_environment_check(script_path):
    """Test that the script checks for the correct environment."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for environment checks
    assert "venv" in content.lower()
    assert "activate" in content.lower()
    assert "python" in content.lower()

def test_script_output_handling(script_path):
    """Test that the script handles output correctly."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for output handling
    assert "write-host" in content.lower()
    assert "write-output" in content.lower()
    assert "write-error" in content.lower()

def test_script_error_handling(script_path):
    """Test that the script handles errors correctly."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for error handling
    assert "try" in content.lower()
    assert "catch" in content.lower()
    assert "error" in content.lower() 