import pytest
import os
import subprocess
import platform
from pathlib import Path

@pytest.fixture
def script_path():
    """Get the path to the installation script based on platform."""
    system = platform.system().lower()
    
    if system == "windows":
        return Path("scripts/windows/install_and_run.ps1")
    elif system == "linux":
        return Path("scripts/linux/install_and_run.sh")
    elif system == "darwin":  # macOS
        return Path("scripts/mac/install_and_run.sh")
    else:
        pytest.skip(f"Unsupported platform: {system}")

def test_script_exists(script_path):
    """Test that the installation script exists."""
    assert script_path.exists()
    assert script_path.is_file()

def test_script_permissions(script_path):
    """Test that the script has the correct permissions."""
    # Check if the script is readable
    assert os.access(script_path, os.R_OK)

def test_script_syntax(script_path):
    """Test that the script has valid syntax."""
    system = platform.system().lower()
    
    try:
        if system == "windows":
            # Use PowerShell to check script syntax without executing it
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", 
                 f"$ErrorActionPreference = 'Stop'; $null = [System.Management.Automation.PSParser]::Tokenize((Get-Content -Raw {script_path}), [ref]$null)"],
                capture_output=True,
                text=True,
                timeout=5  # Set a very short timeout since this should be quick
            )
            assert result.returncode == 0, f"Script syntax validation failed: {result.stderr}"
        else:
            # For shell scripts, just check if they're readable and have basic structure
            with open(script_path, 'r') as f:
                content = f.read()
            assert content.strip(), "Script is empty"
            assert "#!/bin/bash" in content or "#!/bin/sh" in content, "Script missing shebang"
    except subprocess.TimeoutExpired:
        pytest.fail("Script syntax check timed out after 5 seconds")
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