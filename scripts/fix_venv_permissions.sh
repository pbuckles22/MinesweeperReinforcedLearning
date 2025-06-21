#!/bin/bash

# Fix Virtual Environment Permissions Script
# Run this if you have permission issues with your virtual environment

echo "🔧 Fixing Virtual Environment Permissions"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the install script first:"
    echo "  ./scripts/mac/install_and_run.sh"
    exit 1
fi

echo "📁 Found virtual environment: venv/"

# Fix permissions for activation scripts
echo "🔧 Fixing activation script permissions..."
chmod +x venv/bin/activate venv/bin/activate.csh venv/bin/activate.fish 2>/dev/null || true

# Fix permissions for Python executables
echo "🐍 Fixing Python executable permissions..."
chmod +x venv/bin/python* 2>/dev/null || true

# Fix permissions for pip executables
echo "📦 Fixing pip executable permissions..."
chmod +x venv/bin/pip* 2>/dev/null || true

# Fix permissions for all other executables
echo "⚙️  Fixing other executable permissions..."
chmod +x venv/bin/* 2>/dev/null || true

echo "✅ Virtual environment permissions fixed!"
echo ""
echo "🔍 Verifying permissions..."
ls -la venv/bin/ | grep -E "(activate|python|pip)" | head -5

echo ""
echo "🎯 You can now activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "💡 If you still have issues, try:"
echo "   1. Delete the venv directory: rm -rf venv"
echo "   2. Reinstall: ./scripts/mac/install_and_run.sh" 