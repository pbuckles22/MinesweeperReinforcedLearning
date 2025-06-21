#!/bin/bash

# GPU Performance Test Script
# Run this to check if your GPU is performing optimally

echo "🔧 GPU Performance Test"
echo "======================"
echo ""

# Function to test M1 GPU performance
test_m1_performance() {
    echo "🍎 Testing Apple M1 GPU (MPS) Performance..."
    
    # Test matrix multiplication speed
    M1_PERF=$(python -c "
import torch
import time
import numpy as np

device = torch.device('mps')
print(f'Device: {device}')

# Warm up
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
for _ in range(3):
    _ = torch.mm(x, y)

# Benchmark
times = []
for _ in range(20):
    start = time.time()
    z = torch.mm(x, y)
    torch.mps.synchronize()  # Ensure completion
    end = time.time()
    times.append(end - start)

avg_time = np.mean(times)
std_time = np.std(times)
print(f'Matrix multiplication: {avg_time:.3f}s ± {std_time:.3f}s per operation')

# Performance assessment
if avg_time < 0.05:
    print('🚀 EXCELLENT: M1 GPU performing at peak speed')
elif avg_time < 0.1:
    print('✅ GOOD: M1 GPU performing well')
elif avg_time < 0.2:
    print('⚠️  FAIR: M1 GPU performance could be better')
else:
    print('🚨 POOR: M1 GPU performance issues - check thermal throttling')

print(f'Expected range: 0.02-0.15s per operation')
")

    echo "$M1_PERF"
    echo ""
}

# Function to test CUDA GPU performance
test_cuda_performance() {
    echo "🟢 Testing NVIDIA GPU (CUDA) Performance..."
    
    # Get GPU info
    GPU_INFO=$(python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'CUDA Version: {torch.version.cuda}')
")

    echo "$GPU_INFO"
    
    # Test matrix multiplication speed
    CUDA_PERF=$(python -c "
import torch
import time
import numpy as np

device = torch.device('cuda')
print(f'Device: {device}')

# Warm up
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
for _ in range(3):
    _ = torch.mm(x, y)

# Benchmark
times = []
for _ in range(20):
    start = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()  # Ensure completion
    end = time.time()
    times.append(end - start)

avg_time = np.mean(times)
std_time = np.std(times)
print(f'Matrix multiplication: {avg_time:.3f}s ± {std_time:.3f}s per operation')

# Performance assessment
if avg_time < 0.02:
    print('🚀 EXCELLENT: CUDA GPU performing at peak speed')
elif avg_time < 0.05:
    print('✅ GOOD: CUDA GPU performing well')
elif avg_time < 0.1:
    print('⚠️  FAIR: CUDA GPU performance could be better')
else:
    print('🚨 POOR: CUDA GPU performance issues - check utilization')

print(f'Expected range: 0.01-0.08s per operation')
")

    echo "$CUDA_PERF"
    echo ""
}

# Function to test CPU performance
test_cpu_performance() {
    echo "🖥️  Testing CPU Performance..."
    
    CPU_PERF=$(python -c "
import torch
import time
import numpy as np

device = torch.device('cpu')
print(f'Device: {device}')

# Warm up
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
for _ in range(3):
    _ = torch.mm(x, y)

# Benchmark
times = []
for _ in range(20):
    start = time.time()
    z = torch.mm(x, y)
    end = time.time()
    times.append(end - start)

avg_time = np.mean(times)
std_time = np.std(times)
print(f'Matrix multiplication: {avg_time:.3f}s ± {std_time:.3f}s per operation')

# Performance assessment
if avg_time < 0.1:
    print('✅ GOOD: CPU performing well')
elif avg_time < 0.3:
    print('⚠️  FAIR: CPU performance as expected')
else:
    print('🚨 SLOW: CPU performance issues')

print(f'Expected range: 0.05-0.5s per operation')
")

    echo "$CPU_PERF"
    echo ""
}

# Main execution
if python -c "import torch; print('MPS available:', torch.backends.mps.is_available() and torch.backends.mps.is_built())" 2>/dev/null | grep -q "True"; then
    test_m1_performance
elif python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    test_cuda_performance
else
    test_cpu_performance
fi

echo "💡 Performance Tips:"
echo "   - M1 Mac: Check Activity Monitor for thermal throttling"
echo "   - NVIDIA: Check nvidia-smi for GPU utilization"
echo "   - CPU: Close other applications to free up resources"
echo "   - All: Ensure good ventilation and cooling"
echo ""
echo "✅ Performance test complete!" 