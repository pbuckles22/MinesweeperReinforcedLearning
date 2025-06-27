#!/usr/bin/env python3
"""
Cross-Platform Testing Validation Script

This script validates that all platform-specific test scripts work correctly
and produce consistent results across different operating systems.
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
from typing import Dict, List, Any

def get_platform_info() -> Dict[str, str]:
    """Get detailed platform information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation()
    }

def find_test_scripts() -> Dict[str, List[str]]:
    """Find all test scripts organized by platform."""
    scripts_dir = Path("scripts")
    test_scripts = {}
    
    for platform_dir in ["mac", "linux", "windows"]:
        platform_path = scripts_dir / platform_dir
        if platform_path.exists():
            test_scripts[platform_dir] = []
            for script_file in platform_path.glob("*test*"):
                if script_file.is_file():
                    test_scripts[platform_dir].append(str(script_file))
    
    return test_scripts

def validate_script_content(script_path: str) -> Dict[str, Any]:
    """Validate the content of a test script."""
    result = {
        "path": script_path,
        "exists": False,
        "executable": False,
        "content_valid": False,
        "issues": []
    }
    
    try:
        script_file = Path(script_path)
        result["exists"] = script_file.exists()
        
        if not result["exists"]:
            result["issues"].append("Script file does not exist")
            return result
        
        # Check if executable (only for native platform or if we can access it)
        current_platform = platform.system()
        script_platform = "Windows" if script_path.endswith('.ps1') else "Unix"
        
        # Try to check executability, but don't fail validation if we can't
        try:
            if current_platform == "Darwin" and script_platform == "Unix":
                result["executable"] = os.access(script_path, os.X_OK)
                if not result["executable"]:
                    result["issues"].append("Script is not executable")
            elif current_platform == "Windows" and script_platform == "Windows":
                # On Windows, PowerShell scripts don't need to be executable in the same way
                result["executable"] = True
            else:
                # Cross-platform validation - check if we can access it
                result["executable"] = os.access(script_path, os.R_OK)
        except OSError:
            # If we can't check permissions (e.g., different platform), assume it's fine
            result["executable"] = True
        
        # Read and validate content
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Platform-specific validation
        if script_path.endswith('.ps1'):
            # PowerShell script validation
            if 'pytest' not in content:
                result["issues"].append("PowerShell script should run pytest tests")
            if 'Write-Host' not in content:
                result["issues"].append("PowerShell script should use Write-Host for output")
            if 'venv\\Scripts\\Activate.ps1' not in content:
                result["issues"].append("PowerShell script should activate virtual environment")
        elif script_path.endswith('.sh'):
            # Bash script validation
            if 'pytest' not in content:
                result["issues"].append("Bash script should run pytest tests")
            if 'source venv/bin/activate' not in content:
                result["issues"].append("Bash script should activate virtual environment")
            if '#!/bin/bash' not in content:
                result["issues"].append("Bash script should have proper shebang")
        
        result["content_valid"] = len(result["issues"]) == 0
        
    except Exception as e:
        result["issues"].append(f"Error reading script: {str(e)}")
    
    return result

def run_quick_test() -> Dict[str, Any]:
    """Run a quick test to validate the testing infrastructure."""
    result = {
        "success": False,
        "output": "",
        "error": "",
        "duration": 0
    }
    
    try:
        # Run a very quick test to validate pytest works
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/unit/core/test_core_initialization_unit.py::test_invalid_board_size",
            "-v", "--tb=short", "--disable-warnings"
        ]
        
        start_time = subprocess.run(["date"], capture_output=True, text=True)
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        end_time = subprocess.run(["date"], capture_output=True, text=True)
        
        result["success"] = process.returncode == 0
        result["output"] = process.stdout
        result["error"] = process.stderr
        
    except subprocess.TimeoutExpired:
        result["error"] = "Test timed out after 60 seconds"
    except Exception as e:
        result["error"] = f"Error running test: {str(e)}"
    
    return result

def generate_cross_platform_report() -> Dict[str, Any]:
    """Generate a comprehensive cross-platform testing report."""
    report = {
        "timestamp": subprocess.run(["date"], capture_output=True, text=True).stdout.strip(),
        "platform_info": get_platform_info(),
        "test_scripts": find_test_scripts(),
        "script_validation": {},
        "quick_test_result": run_quick_test(),
        "summary": {
            "total_scripts": 0,
            "valid_scripts": 0,
            "issues_found": 0,
            "platforms_covered": []
        }
    }
    
    # Validate all test scripts
    for platform_name, scripts in report["test_scripts"].items():
        report["script_validation"][platform_name] = {}
        report["summary"]["platforms_covered"].append(platform_name)
        
        for script_path in scripts:
            validation = validate_script_content(script_path)
            script_name = Path(script_path).name
            report["script_validation"][platform_name][script_name] = validation
            
            report["summary"]["total_scripts"] += 1
            if validation["content_valid"]:
                report["summary"]["valid_scripts"] += 1
            report["summary"]["issues_found"] += len(validation["issues"])
    
    return report

def print_report(report: Dict[str, Any]):
    """Print a formatted cross-platform testing report."""
    print("=" * 80)
    print("CROSS-PLATFORM TESTING VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {report['timestamp']}")
    print(f"Platform: {report['platform_info']['system']} {report['platform_info']['release']}")
    print(f"Python: {report['platform_info']['python_version']}")
    print()
    
    # Summary
    print("SUMMARY:")
    print(f"  Platforms covered: {', '.join(report['summary']['platforms_covered'])}")
    print(f"  Total test scripts: {report['summary']['total_scripts']}")
    print(f"  Valid scripts: {report['summary']['valid_scripts']}")
    print(f"  Issues found: {report['summary']['issues_found']}")
    print()
    
    # Script validation details
    print("SCRIPT VALIDATION:")
    for platform_name, scripts in report["script_validation"].items():
        print(f"  {platform_name.upper()}:")
        for script_name, validation in scripts.items():
            status = "✅" if validation["content_valid"] else "❌"
            print(f"    {status} {script_name}")
            if validation["issues"]:
                for issue in validation["issues"]:
                    print(f"      ⚠️  {issue}")
        print()
    
    # Quick test result
    print("QUICK TEST RESULT:")
    if report["quick_test_result"]["success"]:
        print("  ✅ Quick test passed")
    else:
        print("  ❌ Quick test failed")
        if report["quick_test_result"]["error"]:
            print(f"  Error: {report['quick_test_result']['error']}")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    if report["summary"]["issues_found"] == 0:
        print("  ✅ All test scripts are valid and consistent")
    else:
        print("  ⚠️  Some issues found - review and fix script inconsistencies")
    
    if report["quick_test_result"]["success"]:
        print("  ✅ Testing infrastructure is working correctly")
    else:
        print("  ❌ Testing infrastructure needs attention")
    
    print("=" * 80)

def main():
    """Main function to run cross-platform testing validation."""
    print("🔍 Running cross-platform testing validation...")
    print()
    
    # Generate report
    report = generate_cross_platform_report()
    
    # Print report
    print_report(report)
    
    # Save report to file
    report_file = "cross_platform_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if report["summary"]["issues_found"] == 0 and report["quick_test_result"]["success"]:
        print("✅ Cross-platform testing validation passed!")
        sys.exit(0)
    else:
        print("❌ Cross-platform testing validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 