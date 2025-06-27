#!/usr/bin/env python3
"""
Periodic Validation Script

This script runs comprehensive validation checks periodically to ensure:
- Cross-platform compatibility is maintained
- Code coverage trends are tracked
- Test quality is monitored
- Performance benchmarks are recorded

Designed to be run weekly/monthly via cron or CI/CD pipeline.

Usage:
    python scripts/periodic_validation.py --frequency weekly
    python scripts/periodic_validation.py --frequency monthly --full-report
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

def get_system_info() -> Dict[str, str]:
    """Get comprehensive system information."""
    import platform
    
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "timestamp": datetime.datetime.now().isoformat()
    }

def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive test suite."""
    print("🧪 Running comprehensive test suite...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest", "tests/",
                "-v", "--tb=short", "--durations=10",
                "--junitxml=test_reports/junit.xml"
            ],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )
        
        duration = time.time() - start_time
        
        # Parse test results
        test_summary = {}
        if result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'passed' in line and 'failed' in line:
                    # Extract test counts
                    parts = line.split()
                    for part in parts:
                        if part.endswith('passed'):
                            test_summary['passed'] = int(part.replace('passed', ''))
                        elif part.endswith('failed'):
                            test_summary['failed'] = int(part.replace('failed', ''))
                        elif part.endswith('skipped'):
                            test_summary['skipped'] = int(part.replace('skipped', ''))
        
        return {
            "success": result.returncode == 0,
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "test_summary": test_summary
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration": 1800,
            "returncode": -1,
            "stdout": "",
            "stderr": "Comprehensive tests timed out after 30 minutes",
            "test_summary": {}
        }
    except Exception as e:
        return {
            "success": False,
            "duration": time.time() - start_time,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Error running comprehensive tests: {str(e)}",
            "test_summary": {}
        }

def run_cross_platform_validation() -> Dict[str, Any]:
    """Run cross-platform validation."""
    print("🔍 Running cross-platform validation...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/validate_cross_platform_tests.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        duration = time.time() - start_time
        
        return {
            "success": result.returncode == 0,
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration": 300,
            "returncode": -1,
            "stdout": "",
            "stderr": "Cross-platform validation timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "duration": time.time() - start_time,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Error running cross-platform validation: {str(e)}"
        }

def run_coverage_analysis() -> Dict[str, Any]:
    """Run comprehensive coverage analysis."""
    print("📊 Running coverage analysis...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/coverage_analysis.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour
        )
        
        duration = time.time() - start_time
        
        # Extract coverage data if available
        coverage_data = {}
        coverage_file = Path("htmlcov/coverage.json")
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
            except Exception:
                pass
        
        return {
            "success": result.returncode == 0,
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "coverage_data": coverage_data
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration": 3600,
            "returncode": -1,
            "stdout": "",
            "stderr": "Coverage analysis timed out after 1 hour",
            "coverage_data": {}
        }
    except Exception as e:
        return {
            "success": False,
            "duration": time.time() - start_time,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Error running coverage analysis: {str(e)}",
            "coverage_data": {}
        }

def run_performance_benchmarks() -> Dict[str, Any]:
    """Run performance benchmarks."""
    print("⚡ Running performance benchmarks...")
    
    benchmarks = {}
    
    # Test environment initialization performance
    try:
        start_time = time.time()
        result = subprocess.run(
            [
                sys.executable, "-c",
                "from src.core.minesweeper_env import MinesweeperEnv; env = MinesweeperEnv(4, 4, 3); print('OK')"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        env_init_time = time.time() - start_time
        
        benchmarks["env_initialization"] = {
            "success": result.returncode == 0,
            "duration": env_init_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        benchmarks["env_initialization"] = {
            "success": False,
            "duration": 0,
            "stdout": "",
            "stderr": str(e)
        }
    
    # Test step performance
    try:
        start_time = time.time()
        result = subprocess.run(
            [
                sys.executable, "-c",
                "from src.core.minesweeper_env import MinesweeperEnv; import time; env = MinesweeperEnv(4, 4, 3); start = time.time(); [env.step(0) for _ in range(100)]; print(f'OK: {time.time() - start:.3f}s')"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        step_time = time.time() - start_time
        
        benchmarks["step_performance"] = {
            "success": result.returncode == 0,
            "duration": step_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        benchmarks["step_performance"] = {
            "success": False,
            "duration": 0,
            "stdout": "",
            "stderr": str(e)
        }
    
    return benchmarks

def load_historical_data() -> Dict[str, Any]:
    """Load historical validation data."""
    history_file = Path("test_reports/validation_history.json")
    
    if history_file.exists():
        try:
            with open(history_file) as f:
                return json.load(f)
        except Exception:
            pass
    
    return {"validations": []}

def save_validation_data(data: Dict[str, Any]):
    """Save validation data to history."""
    history_file = Path("test_reports/validation_history.json")
    
    # Ensure directory exists
    history_file.parent.mkdir(exist_ok=True)
    
    # Load existing history
    history = load_historical_data()
    
    # Add new validation
    history["validations"].append(data)
    
    # Keep only last 50 validations
    if len(history["validations"]) > 50:
        history["validations"] = history["validations"][-50:]
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

def generate_trend_analysis(history: Dict[str, Any]) -> Dict[str, Any]:
    """Generate trend analysis from historical data."""
    if not history.get("validations"):
        return {"trends": "No historical data available"}
    
    validations = history["validations"]
    
    # Calculate trends
    trends = {
        "total_validations": len(validations),
        "success_rate": 0,
        "avg_test_duration": 0,
        "avg_coverage": 0,
        "recent_performance": {}
    }
    
    # Success rate
    successful = sum(1 for v in validations if v.get("tests", {}).get("success", False))
    trends["success_rate"] = successful / len(validations) if validations else 0
    
    # Average test duration
    durations = [v.get("tests", {}).get("duration", 0) for v in validations if v.get("tests", {}).get("duration")]
    trends["avg_test_duration"] = sum(durations) / len(durations) if durations else 0
    
    # Average coverage
    coverages = []
    for v in validations:
        coverage_data = v.get("coverage", {}).get("coverage_data", {})
        if coverage_data:
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
            coverages.append(total_coverage)
    
    trends["avg_coverage"] = sum(coverages) / len(coverages) if coverages else 0
    
    # Recent performance (last 5 validations)
    recent = validations[-5:] if len(validations) >= 5 else validations
    trends["recent_performance"] = {
        "success_rate": sum(1 for v in recent if v.get("tests", {}).get("success", False)) / len(recent),
        "avg_duration": sum(v.get("tests", {}).get("duration", 0) for v in recent) / len(recent)
    }
    
    return trends

def generate_periodic_report(validation_data: Dict[str, Any], trends: Dict[str, Any], frequency: str) -> str:
    """Generate comprehensive periodic report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Periodic Validation Report
**Frequency**: {frequency.title()}
**Generated**: {timestamp}
**Platform**: {validation_data['system_info']['platform']} {validation_data['system_info']['release']}

## Executive Summary
- **Overall Status**: {'✅ PASSED' if validation_data['overall_success'] else '❌ FAILED'}
- **Test Success Rate**: {trends.get('success_rate', 0):.1%}
- **Average Coverage**: {trends.get('avg_coverage', 0):.1f}%
- **Average Test Duration**: {trends.get('avg_test_duration', 0):.1f}s

## Detailed Results

### Test Execution
- **Status**: {'✅ PASSED' if validation_data['tests']['success'] else '❌ FAILED'}
- **Duration**: {validation_data['tests']['duration']:.1f}s
- **Tests Passed**: {validation_data['tests']['test_summary'].get('passed', 0)}
- **Tests Failed**: {validation_data['tests']['test_summary'].get('failed', 0)}
- **Tests Skipped**: {validation_data['tests']['test_summary'].get('skipped', 0)}

### Cross-Platform Validation
- **Status**: {'✅ PASSED' if validation_data['cross_platform']['success'] else '❌ FAILED'}
- **Duration**: {validation_data['cross_platform']['duration']:.1f}s

### Coverage Analysis
- **Status**: {'✅ PASSED' if validation_data['coverage']['success'] else '❌ FAILED'}
- **Duration**: {validation_data['coverage']['duration']:.1f}s
- **Coverage Data**: {'Available' if validation_data['coverage']['coverage_data'] else 'Not Available'}

### Performance Benchmarks
"""
    
    for benchmark_name, benchmark_data in validation_data['benchmarks'].items():
        status = "✅ PASSED" if benchmark_data['success'] else "❌ FAILED"
        report += f"- **{benchmark_name.replace('_', ' ').title()}**: {status} ({benchmark_data['duration']:.3f}s)\n"
    
    report += f"""
## Trend Analysis
- **Total Validations**: {trends.get('total_validations', 0)}
- **Historical Success Rate**: {trends.get('success_rate', 0):.1%}
- **Recent Success Rate**: {trends.get('recent_performance', {}).get('success_rate', 0):.1%}
- **Coverage Trend**: {trends.get('avg_coverage', 0):.1f}%

## Recommendations
"""
    
    if not validation_data['overall_success']:
        report += "- **Immediate Action Required**: Fix failing validations\n"
    
    if trends.get('success_rate', 0) < 0.95:
        report += "- **Quality Concern**: Success rate below 95%\n"
    
    if trends.get('avg_coverage', 0) < 85:
        report += "- **Coverage Concern**: Average coverage below 85%\n"
    
    if trends.get('avg_test_duration', 0) > 300:
        report += "- **Performance Concern**: Test duration exceeds 5 minutes\n"
    
    if validation_data['overall_success'] and trends.get('success_rate', 0) >= 0.95:
        report += "- **All Systems Operational**: Continue development\n"
    
    report += f"""
## Technical Details
- **Python Version**: {validation_data['system_info']['python_version']}
- **Platform**: {validation_data['system_info']['platform']} {validation_data['system_info']['release']}
- **Machine**: {validation_data['system_info']['machine']}
- **Processor**: {validation_data['system_info']['processor']}

---
*Report generated by automated periodic validation system*
"""
    
    return report

def main():
    """Main function for periodic validation."""
    parser = argparse.ArgumentParser(description="Periodic Validation")
    parser.add_argument(
        "--frequency",
        choices=["weekly", "monthly"],
        default="weekly",
        help="Validation frequency"
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Generate full detailed report"
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save results to historical data"
    )
    
    args = parser.parse_args()
    
    # Set PYTHONPATH
    os.environ['PYTHONPATH'] = f"src:{os.environ.get('PYTHONPATH', '')}"
    
    print(f"🚀 Starting {args.frequency} periodic validation...")
    print("=" * 60)
    
    # Run all validations
    validation_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "frequency": args.frequency,
        "system_info": get_system_info(),
        "tests": run_comprehensive_tests(),
        "cross_platform": run_cross_platform_validation(),
        "coverage": run_coverage_analysis(),
        "benchmarks": run_performance_benchmarks()
    }
    
    # Determine overall success
    validation_data["overall_success"] = all([
        validation_data["tests"]["success"],
        validation_data["cross_platform"]["success"],
        validation_data["coverage"]["success"]
    ])
    
    # Load historical data and generate trends
    history = load_historical_data()
    trends = generate_trend_analysis(history)
    
    # Generate report
    report = generate_periodic_report(validation_data, trends, args.frequency)
    
    # Save validation data if requested
    if args.save_history:
        save_validation_data(validation_data)
        print("💾 Validation data saved to history")
    
    # Save report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_reports/periodic_validation_{args.frequency}_{timestamp}.md"
    
    # Ensure reports directory exists
    Path("test_reports").mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 Periodic Validation Summary:")
    print(f"  Overall Status: {'✅ PASSED' if validation_data['overall_success'] else '❌ FAILED'}")
    print(f"  Tests: {'✅ PASSED' if validation_data['tests']['success'] else '❌ FAILED'}")
    print(f"  Cross-Platform: {'✅ PASSED' if validation_data['cross_platform']['success'] else '❌ FAILED'}")
    print(f"  Coverage: {'✅ PASSED' if validation_data['coverage']['success'] else '❌ FAILED'}")
    print(f"  Report: {report_file}")
    
    if args.full_report:
        print("\n" + report)
    
    # Exit with appropriate code
    if validation_data["overall_success"]:
        print("🎉 Periodic validation completed successfully!")
        sys.exit(0)
    else:
        print("⚠️  Periodic validation completed with issues")
        sys.exit(1)

if __name__ == "__main__":
    main() 