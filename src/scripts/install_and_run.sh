#!/bin/bash

# Parse command line arguments
FORCE=false
NO_CACHE=false
USE_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --use-gpu)
            USE_GPU=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if virtual environment exists
if [ -d "venv" ]; then
    if [ "$FORCE" = true ]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        read -p "Virtual environment already exists. Delete it? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        echo "Removing existing virtual environment..."
        rm -rf venv
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
if [ "$NO_CACHE" = true ]; then
    pip install --no-cache-dir -r requirements.txt
else
    pip install -r requirements.txt
fi

# Install PyTorch with CUDA if requested
if [ "$USE_GPU" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Run environment tests
echo "Running environment tests..."
python test_environment.py

# Run training script
echo "Running training script..."
python train_agent.py 