import pytest
import os
import subprocess
from pathlib import Path

@pytest.fixture
def script_path():
    """Get the path to the installation script."""
    return Path("scripts/install_and_run.ps1")

def test_script_exists(script_path):
    """Test that the installation script exists."""
    assert script_path.exists()
    assert script_path.is_file()

def test_script_permissions(script_path):
    """Test that the script has the correct permissions."""
    # Check if the script is executable
    assert os.access(script_path, os.X_OK)

def test_script_syntax(script_path):
    """Test that the script has valid PowerShell syntax."""
    try:
        # Use PowerShell to validate the script syntax
        result = subprocess.run(
            ["powershell", "-Command", f"Get-Content {script_path} | Out-String | Invoke-Expression"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script syntax error: {e.stderr}")

def test_script_dependencies(script_path):
    """Test that the script checks for required dependencies."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for common dependency checks
    assert "python" in content.lower()
    assert "pip" in content.lower()
    assert "venv" in content.lower()

def test_script_environment_setup(script_path):
    """Test that the script sets up the environment correctly."""
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for environment setup steps
    assert "virtualenv" in content.lower() or "venv" in content.lower()
    assert "requirements.txt" in content.lower()
    assert "activate" in content.lower() 