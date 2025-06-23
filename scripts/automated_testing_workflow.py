#!/usr/bin/env python3
"""
Automated Testing Workflow Script

This script provides comprehensive automated testing workflows including:
- Cross-platform validation
- Coverage analysis and reporting
- Test execution and monitoring
- Quality gate enforcement
- Periodic reporting

Usage:
    python scripts/automated_testing_workflow.py --workflow quick
    python scripts/automated_testing_workflow.py --workflow full --coverage
    python scripts/automated_testing_workflow.py --workflow periodic --report
"""

import os
import sys
import json
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_command(cmd: List[str], timeout: int = 300, capture_output: bool = True) -> Dict[str, Any]:
    """Run a command and return results."""
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        end_time = time.time()
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout if capture_output else "",
            "stderr": result.stderr if capture_output else "",
            "duration": end_time - start_time,
            "command": " ".join(cmd)
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "duration": timeout,
            "command": " ".join(cmd)
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Error running command: {str(e)}",
            "duration": 0,
            "command": " ".join(cmd)
        }

def validate_cross_platform_tests() -> Dict[str, Any]:
    """Run cross-platform test validation."""
    print("🔍 Validating cross-platform test consistency...")
    
    result = run_command([
        sys.executable, "scripts/validate_cross_platform_tests.py"
    ])
    
    if result["success"]:
        print("✅ Cross-platform validation passed")
    else:
        print(f"❌ Cross-platform validation failed: {result['stderr']}")
    
    return result

def run_test_suite(suite: str, coverage: bool = False) -> Dict[str, Any]:
    """Run a specific test suite."""
    print(f"🧪 Running test suite: {suite}")
    
    # Split suite into individual paths and filter out empty ones
    suite_paths = [path.strip() for path in suite.split() if path.strip()]
    
    cmd = [sys.executable, "-m", "pytest"] + suite_paths + ["-v", "--tb=short"]
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=json"])
    
    result = run_command(cmd, timeout=600)  # 10 minutes timeout
    
    if result["success"]:
        print(f"✅ Test suite {suite} passed")
    else:
        print(f"❌ Test suite {suite} failed: {result['stderr']}")
    
    return result

def run_coverage_analysis() -> Dict[str, Any]:
    """Run comprehensive coverage analysis."""
    print("📊 Running coverage analysis...")
    
    result = run_command([
        sys.executable, "scripts/coverage_analysis.py"
    ], timeout=1800)  # 30 minutes timeout
    
    if result["success"]:
        print("✅ Coverage analysis completed")
    else:
        print(f"❌ Coverage analysis failed: {result['stderr']}")
    
    return result

def run_quick_coverage() -> Dict[str, Any]:
    """Run quick coverage check."""
    print("⚡ Running quick coverage check...")
    
    result = run_command([
        sys.executable, "scripts/quick_rl_coverage.py"
    ], timeout=300)  # 5 minutes timeout
    
    if result["success"]:
        print("✅ Quick coverage check completed")
    else:
        print(f"❌ Quick coverage check failed: {result['stderr']}")
    
    return result

def generate_test_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive test report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Automated Testing Workflow Report
Generated: {timestamp}

## Summary
- Cross-platform validation: {'✅ PASSED' if results['cross_platform']['success'] else '❌ FAILED'}
- Test execution: {'✅ PASSED' if results['tests']['success'] else '❌ FAILED'}
- Coverage analysis: {'✅ PASSED' if results['coverage']['success'] else '❌ FAILED'}

## Detailed Results

### Cross-Platform Validation
- Duration: {results['cross_platform']['duration']:.2f}s
- Success: {results['cross_platform']['success']}
- Issues: {results['cross_platform']['stderr'] if results['cross_platform']['stderr'] else 'None'}

### Test Execution
- Duration: {results['tests']['duration']:.2f}s
- Success: {results['tests']['success']}
- Issues: {results['tests']['stderr'] if results['tests']['stderr'] else 'None'}

### Coverage Analysis
- Duration: {results['coverage']['duration']:.2f}s
- Success: {results['coverage']['success']}
- Issues: {results['coverage']['stderr'] if results['coverage']['stderr'] else 'None'}

## Recommendations
"""
    
    if not results['cross_platform']['success']:
        report += "- Fix cross-platform test inconsistencies\n"
    
    if not results['tests']['success']:
        report += "- Investigate and fix failing tests\n"
    
    if not results['coverage']['success']:
        report += "- Address coverage analysis issues\n"
    
    if all(result['success'] for result in results.values()):
        report += "- All systems operational - continue development\n"
    
    return report

def save_report(report: str, workflow_type: str):
    """Save the report to a file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_reports/{workflow_type}_workflow_{timestamp}.md"
    
    # Ensure reports directory exists
    Path("test_reports").mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_file}")

def quick_workflow(coverage: bool = False) -> Dict[str, Any]:
    """Run quick testing workflow."""
    print("🚀 Starting Quick Testing Workflow")
    print("=" * 50)
    
    results = {}
    
    # 1. Cross-platform validation
    results['cross_platform'] = validate_cross_platform_tests()
    
    # 2. Quick test suite
    results['tests'] = run_test_suite("tests/unit/core tests/functional/game_flow", coverage)
    
    # 3. Quick coverage if requested
    if coverage:
        results['coverage'] = run_quick_coverage()
    else:
        results['coverage'] = {"success": True, "duration": 0, "stderr": ""}
    
    return results

def full_workflow(coverage: bool = True) -> Dict[str, Any]:
    """Run full testing workflow."""
    print("🚀 Starting Full Testing Workflow")
    print("=" * 50)
    
    results = {}
    
    # 1. Cross-platform validation
    results['cross_platform'] = validate_cross_platform_tests()
    
    # 2. Full test suite
    results['tests'] = run_test_suite("tests/", coverage)
    
    # 3. Comprehensive coverage analysis
    if coverage:
        results['coverage'] = run_coverage_analysis()
    else:
        results['coverage'] = {"success": True, "duration": 0, "stderr": ""}
    
    return results

def periodic_workflow() -> Dict[str, Any]:
    """Run periodic validation workflow."""
    print("🚀 Starting Periodic Validation Workflow")
    print("=" * 50)
    
    results = {}
    
    # 1. Cross-platform validation
    results['cross_platform'] = validate_cross_platform_tests()
    
    # 2. Medium test suite
    results['tests'] = run_test_suite("tests/unit/core tests/functional tests/integration/core")
    
    # 3. Quick coverage check
    results['coverage'] = run_quick_coverage()
    
    return results

def check_quality_gates(results: Dict[str, Any]) -> bool:
    """Check if all quality gates pass."""
    print("\n🔍 Checking Quality Gates...")
    
    gates_passed = 0
    total_gates = len(results)
    
    for gate_name, result in results.items():
        if result['success']:
            print(f"✅ {gate_name}: PASSED")
            gates_passed += 1
        else:
            print(f"❌ {gate_name}: FAILED")
    
    print(f"\nQuality Gates: {gates_passed}/{total_gates} passed")
    
    return gates_passed == total_gates

def main():
    """Main function to run automated testing workflows."""
    parser = argparse.ArgumentParser(description="Automated Testing Workflow")
    parser.add_argument(
        "--workflow", 
        choices=["quick", "full", "periodic"],
        default="quick",
        help="Type of workflow to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Include coverage analysis"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed report"
    )
    parser.add_argument(
        "--quality-gates",
        action="store_true",
        help="Enforce quality gates (fail on any issues)"
    )
    
    args = parser.parse_args()
    
    # Set PYTHONPATH
    os.environ['PYTHONPATH'] = f"src:{os.environ.get('PYTHONPATH', '')}"
    
    print(f"🔧 Running {args.workflow} workflow...")
    print(f"📊 Coverage analysis: {'Enabled' if args.coverage else 'Disabled'}")
    print(f"📄 Report generation: {'Enabled' if args.report else 'Disabled'}")
    print(f"🚦 Quality gates: {'Enabled' if args.quality_gates else 'Disabled'}")
    print()
    
    # Run appropriate workflow
    if args.workflow == "quick":
        results = quick_workflow(args.coverage)
    elif args.workflow == "full":
        results = full_workflow(args.coverage)
    elif args.workflow == "periodic":
        results = periodic_workflow()
    
    # Generate report if requested
    if args.report:
        report = generate_test_report(results)
        save_report(report, args.workflow)
        print("\n" + report)
    
    # Check quality gates if requested
    if args.quality_gates:
        all_passed = check_quality_gates(results)
        if not all_passed:
            print("\n❌ Quality gates failed - workflow terminated")
            sys.exit(1)
        else:
            print("\n✅ All quality gates passed")
    
    # Summary
    total_duration = sum(result.get('duration', 0) for result in results.values())
    print(f"\n⏱️  Total workflow duration: {total_duration:.2f}s")
    
    if all(result['success'] for result in results.values()):
        print("🎉 Workflow completed successfully!")
        sys.exit(0)
    else:
        print("⚠️  Workflow completed with issues")
        sys.exit(1)

if __name__ == "__main__":
    main() 