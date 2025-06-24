#!/usr/bin/env python3
"""
Training Stats History Management Script
Manages training stats files with automatic cleanup and organization.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.train_agent import TrainingStatsManager

def main():
    parser = argparse.ArgumentParser(description='Manage training stats history')
    parser.add_argument('--history-dir', default='training_stats/history', 
                       help='Directory for training stats history (default: training_stats/history)')
    parser.add_argument('--max-age-days', type=int, default=14,
                       help='Maximum age of files to keep in days (default: 14)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to keep (default: None - no limit)')
    parser.add_argument('--action', choices=['summary', 'cleanup', 'list', 'move-existing'],
                       default='summary', help='Action to perform (default: summary)')
    parser.add_argument('--count', type=int, default=5,
                       help='Number of recent files to show (default: 5)')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists("src/core/train_agent.py"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Create stats manager
    stats_manager = TrainingStatsManager(
        history_dir=args.history_dir,
        max_age_days=args.max_age_days,
        max_files=args.max_files
    )
    
    print("📊 Training Stats History Manager")
    print("=" * 40)
    
    if args.action == 'summary':
        # Show summary
        summary = stats_manager.get_stats_summary()
        print(f"📁 History Directory: {summary['history_dir']}")
        print(f"📈 Total Files: {summary['total_files']}")
        print(f"⚙️  Max Age: {args.max_age_days} days")
        print(f"📋 Max Files: {args.max_files if args.max_files else 'No limit'}")
        print(f"🧹 Cleanup includes: training stats, experiment results, and log directories")
        
        if summary['recent_files']:
            print(f"\n📋 Recent Files (last {len(summary['recent_files'])}):")
            for file_info in summary['recent_files']:
                age_str = f"{file_info['age_days']:.1f} days ago"
                size_str = f"{file_info['size']} bytes"
                print(f"   • {file_info['name']} ({age_str}, {size_str})")
        else:
            print("\n📋 No training stats files found.")
    
    elif args.action == 'cleanup':
        # Perform cleanup
        print("🧹 Cleaning up old files...")
        print("   • Training stats files")
        print("   • Experiment result files (modular_results_*.json, simple_results_*.json)")
        print("   • Log directories (benchmark_results, logs/, etc.)")
        stats_manager.cleanup_old_files()
        print("✅ Cleanup completed!")
        
        # Show summary after cleanup
        summary = stats_manager.get_stats_summary()
        print(f"\n📊 After cleanup: {summary['total_files']} files remaining")
    
    elif args.action == 'list':
        # List recent files
        recent_files = stats_manager.get_recent_stats(count=args.count)
        if recent_files:
            print(f"📋 Recent Training Stats Files (last {len(recent_files)}):")
            for file_path in recent_files:
                stat = file_path.stat()
                age_days = (time.time() - stat.st_mtime) / (24 * 60 * 60)
                print(f"   • {file_path.name} ({age_days:.1f} days ago, {stat.st_size} bytes)")
        else:
            print("📋 No training stats files found.")
    
    elif args.action == 'move-existing':
        # Move existing timestamped files to history
        print("📦 Moving existing timestamped training stats to history...")
        moved_count = 0
        
        # Find existing timestamped files in training_stats directory
        training_stats_dir = Path('training_stats')
        if training_stats_dir.exists():
            for file_path in training_stats_dir.glob("training_stats_*.txt"):
                if file_path.name != "training_stats.txt":  # Don't move the current file
                    try:
                        dest_path = stats_manager.move_to_history(str(file_path))
                        if dest_path and dest_path != file_path:
                            print(f"   ✅ Moved: {file_path.name} -> {dest_path}")
                            moved_count += 1
                    except Exception as e:
                        print(f"   ❌ Failed to move {file_path.name}: {e}")
        
        print(f"✅ Moved {moved_count} files to history directory.")
    
    print(f"\n💡 Usage:")
    print(f"   python scripts/manage_training_stats.py --action summary")
    print(f"   python scripts/manage_training_stats.py --action cleanup")
    print(f"   python scripts/manage_training_stats.py --action list --count 10")
    print(f"   python scripts/manage_training_stats.py --action move-existing")

if __name__ == "__main__":
    main() 