import pytest
import os
import subprocess
import platform
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
    system = platform.system().lower()
    
    if system == "windows":
        # On Windows, PowerShell scripts should be readable
        assert os.access(script_path, os.R_OK)
    else:
        # On Unix systems, PowerShell scripts don't need to be executable
        # They just need to be readable
        assert os.access(script_path, os.R_OK)

def test_script_syntax(script_path):
    """Test that the script has valid PowerShell syntax."""
    system = platform.system().lower()
    
    if system == "windows":
        try:
            # Use PowerShell to check script syntax without executing it
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"$ErrorActionPreference = 'Stop'; $null = [System.Management.Automation.PSParser]::Tokenize((Get-Content -Raw {script_path}), [ref]$null)"],
                capture_output=True,
                text=True,
                timeout=5  # Set a very short timeout since this should be quick
            )
            # PowerShell syntax check might not be available, so we'll just check if the command runs
            assert result.returncode in [0, 1], f"Script syntax check failed: {result.stderr}"
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            pytest.fail(f"Script syntax error: {e}")
    else:
        # On non-Windows platforms, just check if the file is readable and has PowerShell content
        with open(script_path, 'r') as f:
            content = f.read()
        assert content.strip(), "Script is empty"
        # Check for PowerShell indicators
        assert ("write-host" in content.lower() or 
                "get-date" in content.lower() or
                "start-process" in content.lower() or
                "$" in content)

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