#!/bin/bash
# Test Version Switcher Script
# Helps switch between different testing versions for comparison

set -e

echo "🧪 Minesweeper RL Test Version Switcher"
echo "========================================"

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Function to switch to a specific test version
switch_to_version() {
    local version=$1
    local branch_name="test/$version"
    
    echo "🔄 Switching to $version version..."
    
    # Check if branch exists locally
    if git show-ref --verify --quiet refs/heads/$branch_name; then
        git checkout $branch_name
    else
        # Try to fetch and checkout from remote
        git fetch origin $branch_name
        git checkout $branch_name
    fi
    
    echo "✅ Switched to $branch_name"
    echo "Current branch: $(git branch --show-current)"
    
    # Show the key difference
    if [ "$version" = "without-learnable-filtering" ]; then
        echo "🔧 Learnable filtering: DISABLED (learnable_only=False)"
    else
        echo "🔧 Learnable filtering: ENABLED (learnable_only=True)"
    fi
}

# Function to run the comprehensive test
run_test() {
    echo "🚀 Running comprehensive DQN test..."
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        echo "📦 Activating virtual environment..."
        source venv/bin/activate
    fi
    
    # Run the test script
    python scripts/training/train_dqn_comprehensive_test.py
}

# Function to show status
show_status() {
    echo "📊 Current Status:"
    echo "   Branch: $(git branch --show-current)"
    echo "   Learnable filtering: $(grep -o 'learnable_only=[^,]*' src/core/minesweeper_env.py | cut -d'=' -f2)"
    echo "   Test script: $(grep -o 'Without Learnable Filtering\|Comprehensive DQN Training Test Script' scripts/training/train_dqn_comprehensive_test.py | head -1)"
}

# Main menu
case "${1:-}" in
    "with")
        switch_to_version "with-learnable-filtering"
        ;;
    "without")
        switch_to_version "without-learnable-filtering"
        ;;
    "test")
        run_test
        ;;
    "status")
        show_status
        ;;
    "both")
        echo "🔄 Running tests on both versions..."
        
        # Test with learnable filtering
        echo "📊 Testing WITH learnable filtering..."
        switch_to_version "with-learnable-filtering"
        run_test
        
        echo ""
        echo "📊 Testing WITHOUT learnable filtering..."
        switch_to_version "without-learnable-filtering"
        run_test
        
        echo "✅ Both tests completed!"
        ;;
    *)
        echo "Usage: $0 {with|without|test|status|both}"
        echo ""
        echo "Commands:"
        echo "  with    - Switch to version WITH learnable filtering"
        echo "  without - Switch to version WITHOUT learnable filtering"
        echo "  test    - Run the comprehensive test on current version"
        echo "  status  - Show current version status"
        echo "  both    - Run tests on both versions sequentially"
        echo ""
        show_status
        ;;
esac 